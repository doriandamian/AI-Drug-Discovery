import logging
import requests
from urllib.parse import quote
from rdkit import Chem, RDLogger
from core.database import get_compound, upsert_compound

logger = logging.getLogger(__name__)

__all__ = ["resolve_smiles"]


def _fetch_smiles_from_pubchem(name: str) -> str | None:
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name, safe='')}/property/IsomericSMILES,ConnectivitySMILES,MolecularWeight,XLogP/JSON"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        props = resp.json()["PropertyTable"]["Properties"][0]
        smiles = (
            props.get("IsomericSMILES")
            or props.get("ConnectivitySMILES")
            or props.get("CanonicalSMILES")
            or props.get("SMILES")
        )
    except Exception:
        logger.debug("PubChem SMILES fetch failed for %r", name, exc_info=True)
        return None

    if not smiles:
        return None
    try:
        upsert_compound(name, {
            "smiles": smiles,
            "mw": props.get("MolecularWeight"),
            "logp": props.get("XLogP"),
        })
    except Exception:
        logger.warning("Neo4j cache write skipped for %r", name, exc_info=True)

    return smiles


def resolve_smiles(raw: str) -> tuple[str | None, object]:
    raw = raw.replace("<smiles>", "").replace("</smiles>", "").strip()

    rdlog = RDLogger.logger()
    rdlog.setLevel(RDLogger.CRITICAL)
    try:
        mol = Chem.MolFromSmiles(raw)
    finally:
        rdlog.setLevel(RDLogger.CRITICAL)

    if mol is not None:
        return raw, mol

    try:
        local = get_compound(raw)
        if local and local.get("smiles"):
            mol = Chem.MolFromSmiles(local["smiles"])
            if mol is not None:
                return local["smiles"], mol
    except Exception:
        logger.debug("Neo4j lookup failed for %r, falling through to PubChem", raw, exc_info=True)

    try:
        smiles = _fetch_smiles_from_pubchem(raw)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return smiles, mol
    except Exception:
        logger.debug("PubChem fallback failed for %r", raw, exc_info=True)

    return None, None
