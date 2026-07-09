import json
import logging
import re

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent as create_react_agent

from tools.pubchem_api import fetch_pubchem_properties, format_pubchem
from tools.toxicity_predictor import predict_toxicity, format_tox_profile
from tools.smiles_validator import validate_smiles, format_validation
from tools.property_calculator import calculate_properties, format_properties
from tools.pubmed_api import search_pubmed
from tools.semantic_scholar_api import search_semantic_scholar
from tools.knowledge_graph import query_knowledge_graph, format_rows
from tools.chembl_api import enrich_drug_graph, format_enrichment
from tools.molecular_design import (
    design_analogs, format_design_report, detect_design_goal, set_design_goal,
)
from rag.retriever import search_literature
from core.config import OLLAMA_BASE_URL, SUBAGENT_MODEL
from agents import trace, smiles_guard

logger = logging.getLogger(__name__)


class SpecialistError(RuntimeError):
    """Raised, not swallowed, so ToolNode surfaces it as an error-status
    message instead of the supervisor composing it into a normal answer.
    """

SUBAGENT_RECURSION_LIMIT = 10

SMILES_GOLDEN_RULE = """\
GOLDEN RULE: a SMILES string is opaque data, never text to rewrite:
- You may ONLY use a SMILES that came out of a tool result.
- You may NEVER type, invent, guess, complete, "correct", or modify a SMILES
  yourself, not even one character.
- ALWAYS pass the compound NAME to tools, never a SMILES string. If a tool says
  "Could not resolve", call fetch_pubchem_properties first, then retry by name."""

OUTPUT_FORMAT = """\
OUTPUT:
- Wrap any SMILES string in <smiles> and </smiles> tags. Never put quotes,
  backticks, or markdown around a SMILES. Print numbers/percentages as plain text.
- Summarize tool results in your own words; never dump raw tool output or JSON.
- Report ONLY values present in a tool result. Never invent data."""


def _make_llm(num_predict: int) -> ChatOllama:
    return ChatOllama(
        model=SUBAGENT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=8192,
        num_predict=num_predict,
        keep_alive=-1,
        reasoning=False,
    )

_llm_subagent = None
_llm_literature = None
_llm_design = None


# Specialist prompts
_CHEMINFORMATICS_PROMPT = f"""You are the Cheminformatics specialist of a drug-discovery system. Your job is to CALL the right tools so a downstream formatter can report a compound's properties and drug-likeness from their structured output. You do NOT write the report or transcribe any numbers yourself.

TOOLS: fetch_pubchem_properties, validate_smiles, calculate_properties.

PROCEDURE: choose the branch that matches the input:

BRANCH A: input is a RAW SMILES STRING (contains characters like =, #, (, ), lowercase letters c/n/o, digits in a chain):
- Do NOT call fetch_pubchem_properties (it takes names, not SMILES).
- Call calculate_properties directly, passing the SMILES string as compound_name. It resolves SMILES natively.
- For a validate-only request on a SMILES, call validate_smiles with the SMILES string.
- After the tool returns, reply with the single word: done.

BRANCH B: input is a COMPOUND NAME (e.g. "aspirin", "ibuprofen"):
- Call fetch_pubchem_properties FIRST (it resolves the real structure and returns CID, MW, logP, and synonyms).
- For a CID / synonym / alternative-name / identity question: call ONLY fetch_pubchem_properties and nothing else, it already contains the CID and synonyms. Do NOT call calculate_properties or validate_smiles for these; they do not return identifiers and would hide the answer.
- MANDATORY: if the question is about drug-likeness, Lipinski's rule of five, QED, functional groups, or computed physicochemical properties, also call calculate_properties after fetch_pubchem_properties, passing the compound NAME.
- For a validate request, call validate_smiles (passing the NAME).
- For a question about SEVERAL compounds, call the needed tool once per compound.
- After the needed tools have returned, reply with the single word: done.

{SMILES_GOLDEN_RULE}"""

_SAFETY_PROMPT = f"""You are the Safety / Toxicology specialist of a drug-discovery system. Your job is to CALL the toxicity screen on the right compound(s); a downstream formatter reports the per-endpoint profile honestly from its structured output. You do NOT write the report or transcribe any probabilities yourself.

TOOLS: fetch_pubchem_properties, predict_toxicity.

PROCEDURE:
- Call fetch_pubchem_properties FIRST for the named drug, THEN predict_toxicity (pass the NAME, never a SMILES).
- For a safety COMPARISON across several compounds, call fetch_pubchem_properties then predict_toxicity for EACH compound, so every one is screened.
- NEVER call fetch_pubchem_properties more than ONCE per compound, even if it returns status "error" (e.g. a biologic/antibody/peptide with no small-molecule record). On failure, proceed straight to predict_toxicity with the compound NAME anyway, it will report its own honest resolution failure, then reply done.
- After predict_toxicity has returned (or failed) for every compound, reply with the single word: done.

{SMILES_GOLDEN_RULE}"""

_LITERATURE_PROMPT = f"""You are the Literature specialist of a drug-discovery system. You retrieve papers and give a SHORT, grounded summary.

TOOLS: search_literature (local pre-loaded knowledge base), search_pubmed (biomedical), search_semantic_scholar (ML / computational, with citation counts). Each returns a JSON document with a "papers" (or "chunks") array of records, read each title / abstract / pmid by its field name.

MANDATORY SEARCH SEQUENCE: you MUST follow all three steps, no exceptions:
1. ALWAYS call search_literature first (local KB).
2. ALWAYS call one web source, this step is NOT optional, even if step 1 returned useful results:
   - search_pubmed for clinical, biomedical, pharmacology, or mechanism questions.
   - search_semantic_scholar for ML, computational-chemistry, or AI drug-discovery questions.
   Stopping after search_literature alone is a grounding failure. You MUST always call at least one web source.
3. Only call the remaining web tool if steps 1-2 were clearly insufficient.

NEVER answer from your own knowledge without completing steps 1-2 first, that is a hallucination.

ANSWER STYLE: this is critical:
- Keep it SHORT: a focused 4-8 sentence summary. NEVER dump or quote whole abstracts.
- Report ONLY what the retrieved abstracts explicitly state. Never infer a mechanism or its direction (what activates/inhibits/increases what) unless an abstract says so in those terms, getting a mechanism backward is worse than saying nothing.
- A paper that merely MENTIONS a drug (e.g. as a test case) does not explain it. If the abstracts do not directly answer the question, say so plainly.
- Attribute claims to their source ("a 2026 study on ... reports ..."). Read across ALL returned papers, not just the first.
- If the user's question contains a factual claim, verify it against the literature before agreeing; never accept the framing as fact.

{OUTPUT_FORMAT}"""

_GRAPH_PROMPT = f"""You are the Knowledge-Graph specialist of a drug-discovery system. Your job is to populate (if needed) and QUERY the local Neo4j graph with READ-ONLY Cypher; a downstream formatter reports the rows your FINAL query returns. You do NOT write the prose answer yourself.

TOOLS: query_knowledge_graph (run Cypher), enrich_drug_graph (add a drug's targets + disease indications from ChEMBL), fetch_pubchem_properties, predict_toxicity.

SCHEMA:
  (:Compound {{name, smiles, molecular_weight, xlogp}})
  (:Compound)-[:HAS_TOXICITY {{probability, cutoff, flagged}}]->(:ToxicityEndpoint {{id}})
  (:Compound)-[:TARGETS {{mechanism, action_type}}]->(:Protein {{chembl_id, name, organism}})
  (:Compound)-[:TREATS {{max_phase}}]->(:Disease {{name, mesh_id}})

SCHEMA LIMITS: data NOT stored (do NOT loop searching for it):
- IC50, Ki, Kd, EC50, or any binding-affinity values are NOT in the schema. If asked for IC50, query what IS stored (mechanism, action_type) and reply done, never retry trying to find affinity data.

CYPHER RULES:
- READ ONLY (MATCH / WHERE / RETURN / ORDER BY / LIMIT). Always end with a LIMIT.
- Compound names are stored capitalized (e.g. 'Aspirin').
- ENTITY NORMALIZATION, proteins/diseases are stored under FULL names, not abbreviations. NEVER match a guessed name with `=`. Use case-insensitive CONTAINS, and expand common abbreviations first:
    COX / COX-1 / COX-2 → 'cyclooxygenase'   (e.g. WHERE toLower(p.name) CONTAINS 'cyclooxygenase')

POPULATE-THEN-QUERY: the graph contains ONLY what earlier tool calls added; it is not a full database:
- You MUST finish with a query_knowledge_graph call that retrieves the data the user asked for, its rows ARE the answer. Enriching alone is NOT enough.
- If a query returns an empty result, the compound is most likely not analysed/enriched yet. Run the populating tool first, enrich_drug_graph for TARGETS/TREATS questions, predict_toxicity (after fetch_pubchem_properties) for HAS_TOXICITY questions, THEN re-run the query.
- EXCEPTION, respect an explicit "do NOT enrich" / "only what is already stored" instruction: in that case call ONLY query_knowledge_graph and do NOT call enrich_drug_graph (or any populating tool), even if the result is empty. An empty result is the correct, honest answer here.
- After your final query_knowledge_graph call, reply with the single word: done.

{SMILES_GOLDEN_RULE}"""

_GRAPH_READONLY_PROMPT = f"""You are the Knowledge-Graph specialist of a drug-discovery system, running in READ-ONLY mode: the user asked for ONLY what is already stored, so you must NOT add data. Your job is to QUERY the local Neo4j graph with READ-ONLY Cypher; a downstream formatter reports the rows your query returns. You do NOT write the prose answer yourself.

TOOL: query_knowledge_graph (run Cypher). You have NO tool to add data, never claim to enrich or populate.

SCHEMA:
  (:Compound {{name, smiles, molecular_weight, xlogp}})
  (:Compound)-[:HAS_TOXICITY {{probability, cutoff, flagged}}]->(:ToxicityEndpoint {{id}})
  (:Compound)-[:TARGETS {{mechanism, action_type}}]->(:Protein {{chembl_id, name, organism}})
  (:Compound)-[:TREATS {{max_phase}}]->(:Disease {{name, mesh_id}})

CYPHER RULES:
- READ ONLY (MATCH / WHERE / RETURN / ORDER BY / LIMIT). Always end with a LIMIT.
- Compound names are stored capitalized (e.g. 'Aspirin').
- ENTITY NORMALIZATION, proteins/diseases are stored under FULL names, not abbreviations. Use case-insensitive CONTAINS and expand abbreviations first (COX / COX-1 / COX-2 → 'cyclooxygenase').

HONEST EMPTY ANSWER: the graph holds ONLY what earlier analyses added; it is NOT a full database:
- Run exactly the query the user asked for, ONCE. If it returns no rows, the compound is simply NOT in the graph yet, that empty result IS the correct, honest answer. Do NOT work around it or add the data.
- After your query_knowledge_graph call, reply with the single word: done.

{SMILES_GOLDEN_RULE}"""

_READONLY_GRAPH_RE = re.compile(
    r"(?:do\s*not|don'?t|without|never|no)\s+(?:first\s+)?(?:enrich|populat)"
    r"|already\s+stored"
    r"|only\s+(?:the\s+|what(?:'s| is| has been)?\s+|already\s+)*stored"
    r"|(?:just|only)\s+query",
    re.IGNORECASE,
)

_MOLECULAR_DESIGN_PROMPT = f"""You are the Molecular Design specialist of a drug-discovery system. Your ONLY job is to RUN the design tool on the right compound; a downstream formatter builds the final report from the tool's structured output, so you do NOT summarize or reformat any scores yourself.

TOOLS: design_analogs (pass the seed compound's NAME), fetch_pubchem_properties.

PROCEDURE:
- Identify the seed compound NAME in the request and call design_analogs with that name DIRECTLY, it resolves the compound itself, so do NOT call fetch_pubchem_properties first.
- ONLY if design_analogs returns status "unresolved", call fetch_pubchem_properties once for that name, then call design_analogs again.
- After design_analogs returns, reply with the single word: done.

{SMILES_GOLDEN_RULE}"""


# Compiled specialist graphs
_cheminformatics = None
_safety = None
_literature = None
_graph = None
_graph_readonly = None
_molecular_design = None


def build_specialists() -> None:
    """Compile all specialist ReAct agents. Must be called after Ollama is reachable."""
    global _llm_subagent, _llm_literature, _llm_design
    global _cheminformatics, _safety, _literature, _graph, _graph_readonly, _molecular_design

    _llm_subagent = _make_llm(num_predict=1024)
    _llm_literature = _make_llm(num_predict=800)
    _llm_design = _make_llm(num_predict=600)

    _cheminformatics = create_react_agent(
        _llm_subagent, tools=[fetch_pubchem_properties, validate_smiles, calculate_properties],
        system_prompt=_CHEMINFORMATICS_PROMPT, name="cheminformatics",
    )
    _safety = create_react_agent(
        _llm_subagent, tools=[fetch_pubchem_properties, predict_toxicity],
        system_prompt=_SAFETY_PROMPT, name="safety",
    )
    _literature = create_react_agent(
        _llm_literature, tools=[search_pubmed, search_semantic_scholar, search_literature],
        system_prompt=_LITERATURE_PROMPT, name="literature",
    )
    _graph = create_react_agent(
        _llm_subagent,
        tools=[query_knowledge_graph, enrich_drug_graph, fetch_pubchem_properties, predict_toxicity],
        system_prompt=_GRAPH_PROMPT, name="graph",
    )
    _graph_readonly = create_react_agent(
        _llm_subagent, tools=[query_knowledge_graph],
        system_prompt=_GRAPH_READONLY_PROMPT, name="graph_readonly",
    )
    _molecular_design = create_react_agent(
        _llm_design, tools=[design_analogs, fetch_pubchem_properties],
        system_prompt=_MOLECULAR_DESIGN_PROMPT, name="molecular_design",
    )


def _stream_agent(agent, task: str) -> tuple[str, list[tuple[str, str]]]:
    """Run one specialist's ReAct loop to completion."""
    final = ""
    inner_tools: list[str] = []
    tool_io: list[tuple[str, str]] = []
    try:
        for chunk in agent.stream(
            {"messages": [("user", task)]},
            stream_mode="updates",
            config={"recursion_limit": SUBAGENT_RECURSION_LIMIT},
        ):
            for _node, payload in chunk.items():
                for msg in payload.get("messages", []):
                    mtype = getattr(msg, "type", None)
                    if getattr(msg, "tool_calls", None):
                        inner_tools.extend(tc["name"] for tc in msg.tool_calls)
                    elif mtype == "tool":
                        content = getattr(msg, "content", "") or ""
                        smiles_guard.record_from_text(content)
                        name = getattr(msg, "name", None)
                        if name:
                            tool_io.append((name, content))
                    elif mtype == "ai" and msg.content:
                        final = msg.content
    except Exception as e:
        name = getattr(agent, "name", "?")
        logger.exception("Specialist '%s' failed", name)
        raise SpecialistError(f"specialist '{name}' failed: {type(e).__name__}: {e}") from e
    finally:
        trace.record(inner_tools)
    return final, tool_io


def _run_agent(agent, task: str) -> str:
    """Run a specialist's ReAct loop and return only its final composed text."""
    final, _ = _stream_agent(agent, task)
    return final or "(the specialist returned no answer)"

_RENDERERS = {
    "fetch_pubchem_properties": format_pubchem,
    "calculate_properties": format_properties,
    "validate_smiles": format_validation,
    "predict_toxicity": format_tox_profile,
    "query_knowledge_graph": format_rows,
    "enrich_drug_graph": format_enrichment,
    "design_analogs": format_design_report,
}


def _render(name: str, raw: str) -> str:
    """Render one tool's JSON output via its deterministic renderer."""
    renderer = _RENDERERS.get(name)
    if renderer is None:
        return raw
    try:
        return renderer(json.loads(raw))
    except (ValueError, TypeError):
        return raw


def _render_from_tools(
    tool_io: list[tuple[str, str]], prefer: tuple[str, ...], fallback: tuple[str, ...] = (),
) -> str:
    """Deterministically render a specialist's answer from its captured tool I/O."""
    blocks = [_render(name, raw) for name, raw in tool_io if name in prefer]
    if not blocks:
        blocks = [_render(name, raw) for name, raw in tool_io if name in fallback]
    return "\n\n".join(b for b in blocks if b)


def _last_output(tool_io: list[tuple[str, str]], *names: str) -> str | None:
    """The raw output of the LAST call to any of `names`."""
    for name, raw in reversed(tool_io):
        if name in names:
            return raw
    return None


def _finish(report: str, fallback: str) -> str:
    """Ground any SMILES in a deterministically-rendered report and return it."""
    if report:
        smiles_guard.record_from_text(report)
        return report
    return fallback or "(the specialist returned no answer)"


def _run_design_agent(agent, task: str) -> str:
    """Run the design specialist, then render its answer deterministically."""
    set_design_goal(detect_design_goal(task))
    try:
        final, tool_io = _stream_agent(agent, task)
    finally:
        set_design_goal("balanced")
    raw = _last_output(tool_io, "design_analogs")
    report = _render("design_analogs", raw) if raw else ""
    return _finish(report, final)


def _run_cheminformatics_agent(agent, task: str) -> str:
    """Property/drug-likeness answer, rendered in code from the tool JSON."""
    final, tool_io = _stream_agent(agent, task)
    report = _render_from_tools(
        tool_io, prefer=("calculate_properties", "validate_smiles"),
        fallback=("fetch_pubchem_properties",),
    )
    return _finish(report, final)

_SAFETY_SCOPE = (
    "Model scope: trained on Tox21 (in-vitro assays) and ClinTox (clinical-trial "
    "failures). It does NOT measure acute lethality (LD50), chronic/repeated-dose "
    "toxicity, carcinogenicity from real use, or addiction. Endpoints not flagged "
    "were simply not flagged by these assays, not proof of safety."
)


def _run_safety_agent(agent, task: str) -> str:
    """Toxicity answer, rendered in code from each predict_toxicity profile."""
    final, tool_io = _stream_agent(agent, task)
    report = _render_from_tools(tool_io, prefer=("predict_toxicity",))
    if report:
        report = f"{report}\n\n{_SAFETY_SCOPE}"
    return _finish(report, final)


def _run_graph_agent(agent, task: str) -> str:
    """Relationship answer, rendered from the FINAL query rows (enrichment is a precursor)."""
    final, tool_io = _stream_agent(agent, task)
    raw = _last_output(tool_io, "query_knowledge_graph")
    name = "query_knowledge_graph"
    if raw is None:
        raw = _last_output(tool_io, "enrich_drug_graph")
        name = "enrich_drug_graph"
    report = _render(name, raw) if raw else ""
    return _finish(report, final)


# Specialist tools exposed to the supervisor
@tool
def cheminformatics_agent(task: str) -> str:
    """Compound IDENTITY, physicochemical PROPERTIES (MW, logP, TPSA...), DRUG-LIKENESS
    (Lipinski, QED). Use for "molecular weight of X", "is X drug-like / passes
    Lipinski", "validate X". Pass a task naming the compound(s)."""
    return _run_cheminformatics_agent(_cheminformatics, task)


@tool
def safety_agent(task: str) -> str:
    """ML TOXICITY screening: per-endpoint profiles, flagged endpoints, safety
    comparisons (with model-scope caveats). Use for "is X toxic", "predict toxicity
    of X", "compare the safety of X and Y". Pass a task naming the compound(s)."""
    return _run_safety_agent(_safety, task)


@tool
def literature_agent(task: str) -> str:
    """Published EVIDENCE (local KB + PubMed / Semantic Scholar): mechanisms,
    pharmacology, clinical findings, ADMET/SAR, verifying a factual claim. Use for
    "what does the literature say about X", "is it true that X causes Y", "find
    papers on X". Pass the full question as the task."""
    return _run_agent(_literature, task)


@tool
def graph_agent(task: str) -> str:
    """Knowledge-graph CROSS-COMPOUND / RELATIONSHIP questions: which drugs target a
    protein, what X targets/treats, which compounds are flagged for an endpoint,
    comparing targets/endpoints across compounds (can enrich then query). Use for
    "which drugs target COX", "what does X target / treat". Pass the full question."""
    agent = _graph_readonly if _READONLY_GRAPH_RE.search(task or "") else _graph
    return _run_graph_agent(agent, task)


@tool
def molecular_design_agent(task: str) -> str:
    """GENERATE & rank novel candidate analogs of a known compound for drug-likeness,
    predicted toxicity, and synthesizability, returns UNVALIDATED computational
    proposals, the ONLY way to create structures. Use for any design / generate /
    optimize / "safer or more drug-like version" request; never invent a SMILES
    yourself. Pass the request naming the seed compound."""
    return _run_design_agent(_molecular_design, task)


def warmup_subagents():
    """Load each distinct sub-agent model into Ollama so the first real delegation
    is not a cold start.  Deduplicates by model name so redundant round-trips are
    skipped even when multiple LLM instances share the same model."""
    seen: set[str] = set()
    for llm in (_llm_subagent, _llm_literature, _llm_design):
        if llm is None:
            continue
        model_name: str = getattr(llm, "model", "") or ""
        if model_name in seen:
            continue
        seen.add(model_name)
        try:
            llm.invoke("ready?")
        except Exception:
            logger.warning("Sub-agent model warmup skipped for %r", model_name, exc_info=True)
