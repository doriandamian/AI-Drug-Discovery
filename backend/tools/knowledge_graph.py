import json
import logging

from langchain_core.tools import tool
from core.database import run_read_query
from tools.entity_resolver import expand_abbreviations, fuzzy_resolve

__all__ = ["query_knowledge_graph", "format_rows"]

logger = logging.getLogger(__name__)

SCHEMA = """\
NODES
  (:Compound {name, smiles, molecular_weight, xlogp})
  (:ToxicityEndpoint {id})   // e.g. 'SR-MMP', 'NR-AR', 'ClinTox'
  (:Protein {chembl_id, name, organism})
  (:Disease {name, mesh_id})
RELATIONSHIPS
  (:Compound)-[:HAS_TOXICITY {probability, cutoff, flagged}]->(:ToxicityEndpoint)
  (:Compound)-[:TARGETS {mechanism, action_type}]->(:Protein)
  (:Compound)-[:TREATS {max_phase}]->(:Disease)
NOTES
  - Compound.name is stored capitalized (e.g. 'Aspirin').
  - r.flagged is a boolean: true when probability >= cutoff.
  - The graph only contains data added by earlier tool calls, it is NOT a
    complete drug database. Empty results mean 'not yet analysed/enriched', not
    'does not exist'."""


@tool(description=f"""Run a READ-ONLY Cypher query against the local Neo4j drug knowledge graph and return the matching rows.

Use this for cross-compound / relationship questions the per-compound tools cannot answer, e.g. "which analysed compounds are flagged for the SR-MMP endpoint?" or "list every toxicity endpoint stored for Aspirin". Independent; can run alongside other tools in any step.

SCHEMA:
{SCHEMA}

RULES:
1. READ ONLY. Never write CREATE, MERGE, SET, DELETE, REMOVE, DROP, LOAD CSV, or CALL apoc.*, these are all rejected.
2. Always end with a LIMIT (e.g. LIMIT 25).
3. Match Compound names capitalized, e.g. {{name: 'Aspirin'}}.
4. If the result is empty, the compound has simply not been analysed yet, say so; do not invent rows.""")
def query_knowledge_graph(cypher: str) -> str:
    cypher = cypher.strip().rstrip(";")
    if not cypher:
        return json.dumps({"status": "error", "message": "empty Cypher query."})
    notes: list[str] = []
    cypher, expanded = expand_abbreviations(cypher)
    if expanded:
        notes.append("normalized abbreviations in the query")

    try:
        rows = run_read_query(cypher)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error executing Cypher: {e}"})

    if not rows:
        try:
            retry_cypher, subs = fuzzy_resolve(cypher)
        except Exception:
            logger.warning("Fuzzy entity resolution failed", exc_info=True)
            retry_cypher, subs = cypher, []
        if subs:
            try:
                rows = run_read_query(retry_cypher)
            except Exception as e:
                return json.dumps({"status": "error", "message": f"Error executing Cypher: {e}"})
            if rows:
                notes.append(
                    "matched "
                    + ", ".join(f"'{orig}'→'{canon}'" for orig, canon in subs)
                )

    if not rows:
        return json.dumps({
            "status": "empty", "row_count": 0, "rows": [], "notes": notes,
            "message": "No rows matched. The referenced compound(s) may not have been "
                       "analysed yet.",
        })

    return json.dumps({
        "status": "ok",
        "row_count": len(rows),
        "rows": [dict(row) for row in rows],
        "notes": notes,
    })


def format_rows(payload: dict) -> str:
    status = payload.get("status")
    notes = payload.get("notes") or []
    prefix = f"(note: {'; '.join(notes)})\n" if notes else ""

    if status == "error":
        return f"Error: {payload.get('message', 'query failed')}"
    if status == "empty" or not payload.get("rows"):
        return prefix + payload.get(
            "message", "No rows matched. The referenced compound(s) may not have been analysed yet."
        )

    rows = payload["rows"]
    lines = [f"{prefix}{len(rows)} row(s):"]
    for row in rows:
        lines.append("  • " + ", ".join(f"{k}={v}" for k, v in row.items()))
    return "\n".join(lines)
