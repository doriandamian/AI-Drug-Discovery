import json
import logging
import threading
import time
from urllib.parse import quote

import requests
from langchain_core.tools import tool
from core.database import upsert_compound

__all__ = ["fetch_pubchem_properties", "format_pubchem"]

logger = logging.getLogger(__name__)

_session = requests.Session()

_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_LOCK = threading.Lock()
_OK_TTL = 3600.0
_MISS_TTL = 120.0


def _norm(name: str) -> str:
    return (name or "").strip().lower()


def _cache_get(name: str) -> str | None:
    key = _norm(name)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is None:
            return None
        expires_at, payload = hit
        if time.monotonic() >= expires_at:
            _CACHE.pop(key, None)
            return None
        return payload


def _cache_put(name: str, payload: str, ttl: float) -> str:
    with _CACHE_LOCK:
        _CACHE[_norm(name)] = (time.monotonic() + ttl, payload)
    return payload


@tool(description="Fetches chemical identity and properties for a named compound from PubChem and saves them to the local database. Input: The exact name of the compound (e.g., 'Aspirin' or 'Ibuprofen'). Returns a structured JSON document with keys: status, compound, cid (PubChem CID integer), molecular_formula, molecular_weight, logp, smiles_stored (bool), synonyms (list of alternative names). Read every value by its KEY. Use this tool to look up a compound's PubChem CID, synonyms/alternative names, MW, or logP. After calling this tool, pass the compound NAME (not a SMILES string) to predict_toxicity, calculate_properties, and validate_smiles.")
def fetch_pubchem_properties(compound_name: str) -> str:
    cached = _cache_get(compound_name)
    if cached is not None:
        return cached

    prop_url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(compound_name, safe='')}/property/"
        f"IsomericSMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,XLogP/JSON"
    )
    try:
        response = _session.get(prop_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        properties = data['PropertyTable']['Properties'][0]

        cid = properties.get('CID')
        smiles = (
            properties.get('IsomericSMILES')
            or properties.get('ConnectivitySMILES')
            or properties.get('CanonicalSMILES')
        )
        clean_props = {
            "smiles": smiles,
            "molecular_formula": properties.get('MolecularFormula'),
            "molecular_weight": properties.get('MolecularWeight'),
            "xlogp": properties.get('XLogP'),
        }

        try:
            upsert_compound(compound_name, clean_props)
        except Exception:
            logger.warning("Neo4j upsert skipped for %r", compound_name, exc_info=True)

        synonyms: list[str] = []
        if cid:
            try:
                syn_url = (
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                    f"{cid}/synonyms/JSON"
                )
                syn_resp = _session.get(syn_url, timeout=8)
                if syn_resp.ok:
                    syn_data = syn_resp.json()
                    raw_syns = (
                        syn_data.get("InformationList", {})
                        .get("Information", [{}])[0]
                        .get("Synonym", [])
                    )
                    synonyms = raw_syns[:8]
            except Exception:
                logger.debug("Synonym fetch skipped for %r", compound_name, exc_info=True)

        return _cache_put(compound_name, json.dumps({
            "status": "ok",
            "compound": compound_name,
            "source": "pubchem",
            "cid": cid,
            "molecular_formula": clean_props['molecular_formula'],
            "molecular_weight": clean_props['molecular_weight'],
            "logp": clean_props['xlogp'],
            "smiles_stored": True,
            "synonyms": synonyms,
        }), _OK_TTL)
    except Exception as e:
        return _cache_put(compound_name, json.dumps({
            "status": "error",
            "compound": compound_name,
            "message": f"Could not find data for '{compound_name}' in PubChem. ({e})",
        }), _MISS_TTL)


def format_pubchem(payload: dict) -> str:
    if payload.get("status") != "ok":
        return payload.get("message", "PubChem lookup failed.")
    formula = payload.get("molecular_formula")
    mw = payload.get("molecular_weight")
    logp = payload.get("logp")
    cid = payload.get("cid")
    synonyms = payload.get("synonyms") or []
    lines = [
        f"Properties for '{payload['compound']}' (source: {payload.get('source', 'pubchem')}):",
        f"- PubChem CID: {cid if cid is not None else 'N/A'}",
        f"- Molecular Formula: {formula if formula is not None else 'N/A'}",
        f"- Molecular Weight: {mw if mw is not None else 'N/A'} g/mol",
        f"- logP (lipophilicity): {logp if logp is not None else 'N/A'}",
    ]
    if synonyms:
        lines.append(f"- Synonyms: {', '.join(synonyms)}")
    return "\n".join(lines)
