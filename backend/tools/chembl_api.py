import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from langchain_core.tools import tool

from core.database import link_targets, link_indications

__all__ = ["enrich_drug_graph", "format_enrichment"]

logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
TIMEOUT = 15
MAX_TARGETS = 10
MAX_DISEASES = 10
_HEADERS = {"User-Agent": "drug_discovery_agent/1.0"}
_TARGET_CHUNK = 50


def _get(path: str, params: dict) -> dict:
    url = f"{CHEMBL_BASE}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _resolve_molecule(name: str) -> str | None:
    data = _get("molecule/search.json", {"q": name, "limit": 1})
    molecules = data.get("molecules", [])
    return molecules[0].get("molecule_chembl_id") if molecules else None


def _fetch_targets(molecule_id: str) -> list[dict]:
    data = _get("mechanism.json", {"molecule_chembl_id": molecule_id, "limit": MAX_TARGETS})
    mechanisms = data.get("mechanisms", [])
    if not mechanisms:
        return []

    seen: dict[str, dict] = {}
    for mech in mechanisms:
        tid = mech.get("target_chembl_id")
        if tid and tid not in seen:
            seen[tid] = mech

    target_info: dict[str, dict] = {}
    tids = list(seen)
    for i in range(0, len(tids), _TARGET_CHUNK):
        chunk = tids[i : i + _TARGET_CHUNK]
        try:
            resp = _get("target.json", {
                "target_chembl_id__in": ",".join(chunk),
                "limit": len(chunk),
            })
            for t in resp.get("targets", []):
                tid = t.get("target_chembl_id")
                if tid:
                    target_info[tid] = {
                        "name": t.get("pref_name") or tid,
                        "organism": t.get("organism"),
                    }
        except Exception:
            logger.warning(
                "ChEMBL bulk target lookup failed for chunk [%d:%d]",
                i, i + len(chunk),
                exc_info=True,
            )

    return [
        {
            "chembl_id": tid,
            "name": target_info.get(tid, {}).get("name") or tid,
            "organism": target_info.get(tid, {}).get("organism"),
            "mechanism": mech.get("mechanism_of_action"),
            "action_type": mech.get("action_type"),
        }
        for tid, mech in seen.items()
    ]


def _fetch_indications(molecule_id: str) -> list[dict]:
    data = _get("drug_indication.json", {"molecule_chembl_id": molecule_id, "limit": MAX_DISEASES})
    diseases = []
    seen = set()
    for ind in data.get("drug_indications", []):
        disease = ind.get("mesh_heading") or ind.get("efo_term")
        if not disease or disease in seen:
            continue
        seen.add(disease)
        diseases.append({
            "name": disease,
            "mesh_id": ind.get("mesh_id"),
            "max_phase": ind.get("max_phase_for_ind"),
        })
    return diseases


@tool(description="""Enriches the knowledge graph with a drug's molecular TARGETS (mechanism of action) and approved DISEASE indications, fetched from the ChEMBL database, then stores them as (:Compound)-[:TARGETS]->(:Protein) and (:Compound)-[:TREATS]->(:Disease).

WHEN TO USE: the user asks what a drug targets, how it works mechanistically, what proteins/enzymes/receptors it acts on, or what conditions it treats, OR before a query_knowledge_graph question that needs target/disease relationships for a compound not yet enriched.

INPUT (compound_name): the drug name, e.g. 'Aspirin', 'Imatinib'. Pass the name, never a SMILES string.

After this runs, use query_knowledge_graph to answer relational questions across the enriched compounds. The data is real and sourced, report targets/diseases only as returned, do not invent mechanisms.""")
def enrich_drug_graph(compound_name: str) -> str:
    try:
        molecule_id = _resolve_molecule(compound_name)
    except Exception as e:
        logger.exception("ChEMBL molecule resolution failed")
        return json.dumps({"status": "error", "compound": compound_name,
                           "message": f"Could not reach ChEMBL to resolve '{compound_name}' ({e})."})

    if not molecule_id:
        return json.dumps({"status": "not_found", "compound": compound_name,
                           "message": f"Could not find '{compound_name}' in ChEMBL. No targets or "
                                      f"indications to add."})

    try:
        targets = _fetch_targets(molecule_id)
    except Exception:
        logger.warning("ChEMBL mechanism fetch failed", exc_info=True)
        targets = []
    try:
        diseases = _fetch_indications(molecule_id)
    except Exception:
        logger.warning("ChEMBL indication fetch failed", exc_info=True)
        diseases = []

    if not targets and not diseases:
        return json.dumps({"status": "empty", "compound": compound_name, "chembl_id": molecule_id,
                           "targets": [], "indications": [],
                           "message": f"'{compound_name}' ({molecule_id}) is in ChEMBL but has no "
                                      f"recorded mechanism-of-action targets or indications. Nothing "
                                      f"added to the graph."})
    try:
        link_targets(compound_name, targets)
        link_indications(compound_name, diseases)
    except Exception:
        logger.warning("Knowledge-graph write failed during enrichment", exc_info=True)

    return json.dumps({
        "status": "ok",
        "compound": compound_name,
        "chembl_id": molecule_id,
        "targets": targets,
        "indications": diseases,
    })


def format_enrichment(payload: dict) -> str:
    if payload.get("status") in ("error", "not_found", "empty"):
        return payload.get("message", "Enrichment produced no data.")

    lines = [f"Enriched '{payload['compound']}' (ChEMBL {payload['chembl_id']}) into the "
             f"knowledge graph:"]
    targets = payload.get("targets") or []
    diseases = payload.get("indications") or []
    if targets:
        lines.append(f"TARGETS ({len(targets)}):")
        for t in targets:
            action = t.get("action_type") or "acts on"
            lines.append(f"  • {t['name']}, {action}: {t.get('mechanism') or 'n/a'}")
    else:
        lines.append("TARGETS: none recorded.")
    if diseases:
        lines.append(f"INDICATIONS ({len(diseases)}):")
        for d in diseases:
            phase = d.get("max_phase")
            phase_str = f" (max phase {phase})" if phase is not None else ""
            lines.append(f"  • {d['name']}{phase_str}")
    else:
        lines.append("INDICATIONS: none recorded.")
    return "\n".join(lines)
