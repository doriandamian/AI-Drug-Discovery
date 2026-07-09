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
from agents.prompts import (
    CHEMINFORMATICS_PROMPT as _CHEMINFORMATICS_PROMPT,
    SAFETY_PROMPT as _SAFETY_PROMPT,
    LITERATURE_PROMPT as _LITERATURE_PROMPT,
    GRAPH_PROMPT as _GRAPH_PROMPT,
    GRAPH_READONLY_PROMPT as _GRAPH_READONLY_PROMPT,
    MOLECULAR_DESIGN_PROMPT as _MOLECULAR_DESIGN_PROMPT,
)

logger = logging.getLogger(__name__)


class SpecialistError(RuntimeError):
    """Raised, not swallowed, so ToolNode surfaces it as an error-status
    message instead of the supervisor composing it into a normal answer.
    """

SUBAGENT_RECURSION_LIMIT = 10


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


_READONLY_GRAPH_RE = re.compile(
    r"(?:do\s*not|don'?t|without|never|no)\s+(?:first\s+)?(?:enrich|populat)"
    r"|already\s+stored"
    r"|only\s+(?:the\s+|what(?:'s| is| has been)?\s+|already\s+)*stored"
    r"|(?:just|only)\s+query",
    re.IGNORECASE,
)


_cheminformatics = None
_safety = None
_literature = None
_graph = None
_graph_readonly = None
_molecular_design = None


def build_specialists() -> None:
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
    renderer = _RENDERERS.get(name)
    if renderer is None:
        return raw
    try:
        return renderer(json.loads(raw))
    except (ValueError, TypeError, KeyError):
        return raw


def _render_from_tools(
    tool_io: list[tuple[str, str]], prefer: tuple[str, ...], fallback: tuple[str, ...] = (),
) -> str:
    blocks = [_render(name, raw) for name, raw in tool_io if name in prefer]
    if not blocks:
        blocks = [_render(name, raw) for name, raw in tool_io if name in fallback]
    return "\n\n".join(b for b in blocks if b)


def _last_output(tool_io: list[tuple[str, str]], *names: str) -> str | None:
    for name, raw in reversed(tool_io):
        if name in names:
            return raw
    return None


def _finish(report: str, fallback: str) -> str:
    if report:
        smiles_guard.record_from_text(report)
        return report
    return fallback or "(the specialist returned no answer)"


def _run_design_agent(agent, task: str) -> str:
    set_design_goal(detect_design_goal(task))
    try:
        final, tool_io = _stream_agent(agent, task)
    finally:
        set_design_goal("balanced")
    raw = _last_output(tool_io, "design_analogs")
    report = _render("design_analogs", raw) if raw else ""
    return _finish(report, final)


def _run_cheminformatics_agent(agent, task: str) -> str:
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
    final, tool_io = _stream_agent(agent, task)
    report = _render_from_tools(tool_io, prefer=("predict_toxicity",))
    if report:
        report = f"{report}\n\n{_SAFETY_SCOPE}"
    return _finish(report, final)


def _run_graph_agent(agent, task: str) -> str:
    final, tool_io = _stream_agent(agent, task)
    raw = _last_output(tool_io, "query_knowledge_graph")
    name = "query_knowledge_graph"
    if raw is None:
        raw = _last_output(tool_io, "enrich_drug_graph")
        name = "enrich_drug_graph"
    report = _render(name, raw) if raw else ""
    return _finish(report, final)


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
