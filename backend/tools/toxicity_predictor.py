import json
import logging
import os
import warnings
import numpy as np
import joblib
from langchain_core.tools import tool
from ml.model_integrity import verify_hash

logger = logging.getLogger(__name__)
from ml.features import featurize
from tools.smiles_resolver import resolve_smiles
from core.database import link_toxicity_endpoints

__all__ = [
    "reload_bundle",
    "get_ad_threshold",
    "safety_oracle_available",
    "predict_endpoint_probs",
    "predict_toxicity",
    "format_tox_profile",
]

warnings.filterwarnings("ignore", message="X does not have valid feature names")

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "toxicity_model.pkl")

_DEFAULTS = {"threshold": 0.5, "ad_fingerprints": None, "ad_threshold": 0.30}


def _load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    verify_hash(MODEL_PATH)
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict):
        return {**_DEFAULTS, **obj}
    return {**_DEFAULTS, "model": obj}


_BUNDLE = _load_bundle()
_MULTITASK = bool(_BUNDLE and _BUNDLE.get("tasks"))
_AD_FPS = (
    np.unpackbits(_BUNDLE["ad_fingerprints"], axis=1).astype(np.uint8)
    if _BUNDLE and _BUNDLE["ad_fingerprints"] is not None else None
)


def reload_bundle():
    global _BUNDLE, _MULTITASK, _AD_FPS
    _BUNDLE = _load_bundle()
    _MULTITASK = bool(_BUNDLE and _BUNDLE.get("tasks"))
    _AD_FPS = (
        np.unpackbits(_BUNDLE["ad_fingerprints"], axis=1).astype(np.uint8)
        if _BUNDLE and _BUNDLE["ad_fingerprints"] is not None else None
    )


def get_ad_threshold() -> float:
    return float(_BUNDLE["ad_threshold"]) if _BUNDLE else float(_DEFAULTS["ad_threshold"])


def safety_oracle_available() -> bool:
    return _MULTITASK


def _max_train_similarity(fp_bits):
    if _AD_FPS is None:
        return None
    q = fp_bits.reshape(1, -1).astype(np.uint8)
    inter = (_AD_FPS & q).sum(axis=1)
    union = (_AD_FPS | q).sum(axis=1)
    union = np.where(union == 0, 1, union)
    return float((inter / union).max())


def _domain_info(fp_bits) -> dict | None:
    sim = _max_train_similarity(fp_bits)
    if sim is None:
        return None
    threshold = float(_BUNDLE["ad_threshold"])
    return {
        "ad_similarity": round(sim, 2),
        "threshold": round(threshold, 2),
        "low_reliability": sim < threshold,
    }


def predict_endpoint_probs(smiles: str) -> dict | None:
    if not _MULTITASK:
        return None
    res = featurize(smiles)
    if res is None:
        return None
    combined, fp_bits = res
    x = combined.reshape(1, -1)
    probs = {
        name: float(t["model"].predict_proba(x)[0][1])
        for name, t in _BUNDLE["tasks"].items()
    }
    return {"probs": probs, "ad_similarity": _max_train_similarity(fp_bits)}


def _predict_single(combined, fp_bits, smiles) -> dict:
    prob_toxic = float(_BUNDLE["model"].predict_proba(combined.reshape(1, -1))[0][1])
    threshold = float(_BUNDLE["threshold"])
    return {
        "status": "ok",
        "model": "single_task",
        "compound": smiles,
        "probability": round(prob_toxic, 4),
        "cutoff": round(threshold, 4),
        "flagged": prob_toxic >= threshold,
        "borderline": abs(prob_toxic - threshold) < 0.10,
        "domain": _domain_info(fp_bits),
    }


def _predict_multitask(combined, fp_bits, smiles, compound_name=None) -> dict:
    info = _BUNDLE.get("task_info", {})
    x = combined.reshape(1, -1)

    results = []
    for name, t in _BUNDLE["tasks"].items():
        prob = float(t["model"].predict_proba(x)[0][1])
        cutoff = t["threshold"]
        flagged = prob >= cutoff
        results.append((name, prob, cutoff, flagged))
    results.sort(key=lambda r: -(r[1] - r[2]))

    if compound_name and compound_name != smiles:
        try:
            link_toxicity_endpoints(compound_name, results)
        except Exception:
            logger.warning("Knowledge-graph toxicity write failed for %r", compound_name, exc_info=True)

    label = compound_name if (compound_name and compound_name != smiles) else smiles
    endpoints = [
        {
            "id": name,
            "description": info.get(name, ""),
            "probability": round(prob, 4),
            "cutoff": round(cutoff, 4),
            "flagged": bool(flagged),
        }
        for name, prob, cutoff, flagged in results
    ]
    top = max(results, key=lambda r: r[1])
    return {
        "status": "ok",
        "model": "multitask",
        "compound": label,
        "n_endpoints": len(endpoints),
        "endpoints": endpoints,
        "flagged": [e for e in endpoints if e["flagged"]],
        "highest_probability": {
            "id": top[0],
            "description": info.get(top[0], ""),
            "probability": round(float(top[1]), 4),
        },
        "domain": _domain_info(fp_bits),
    }


@tool(description="""Predicts a molecule's toxicity using calibrated machine-learning models trained on the Tox21 and ClinTox datasets. Returns a per-endpoint toxicity profile (e.g. endocrine, mitochondrial, DNA-damage) rather than a single yes/no.

INPUT (compound_name): The compound name that was already looked up with fetch_pubchem_properties (e.g. 'Aspirin', 'Bisphenol A'). Always pass the compound name, never a SMILES string.

RULES:
1. Pass the compound name when you already fetched it via fetch_pubchem_properties. Only pass a raw SMILES if you never fetched the compound by name.
2. NEVER invent, guess, or modify a SMILES yourself. If you must pass a SMILES, copy it character-for-character from the tool result.
3. The models are probabilistic screens, not ground truth. Always report the flagged endpoints and any applicability-domain warning.""")
def predict_toxicity(compound_name: str) -> str:
    if _BUNDLE is None or (not _MULTITASK and not _BUNDLE.get("model")):
        return json.dumps({"status": "error", "compound": compound_name,
                           "message": "Toxicity prediction model not found. Train it first "
                                      "(python ml/train_multitask.py)."})

    resolved_smiles, mol = resolve_smiles(compound_name)
    if resolved_smiles is None:
        return json.dumps({"status": "error", "compound": compound_name,
                           "message": f"No small-molecule structure available for '{compound_name}'. "
                                      f"It is likely a biologic (antibody/peptide/protein) or an "
                                      f"unrecognized name; no toxicity screen is available for it. "
                                      f"Do not retry this tool for this compound."})

    result = featurize(resolved_smiles)
    if result is None:
        return json.dumps({"status": "error", "compound": compound_name,
                           "message": f"Invalid SMILES string provided: '{resolved_smiles}'"})
    combined, fp_bits = result

    try:
        if _MULTITASK:
            payload = _predict_multitask(combined, fp_bits, resolved_smiles, compound_name)
        else:
            payload = _predict_single(combined, fp_bits, resolved_smiles)
        if compound_name and compound_name != resolved_smiles:
            payload["compound"] = compound_name
        return json.dumps(payload)
    except Exception as e:
        return json.dumps({"status": "error", "compound": compound_name,
                           "message": f"Error during prediction: {e}"})

def _domain_note(domain: dict | None) -> str:
    if domain and domain.get("low_reliability"):
        return (
            f"\n⚠ Low reliability: this molecule is structurally unlike the training data "
            f"(max similarity {domain['ad_similarity']:.2f} < {domain['threshold']:.2f}). "
            f"Treat the prediction as a weak hint, not evidence."
        )
    return ""


def format_tox_profile(payload: dict) -> str:
    if payload.get("status") != "ok":
        return payload.get("message", "Toxicity prediction failed.")

    if payload.get("model") == "single_task":
        verdict = "Toxic" if payload["flagged"] else "not flagged by this assay"
        border = ", borderline, close to the decision cutoff" if payload["borderline"] else ""
        return (
            f"Toxicity screen for '{payload['compound']}': {verdict}{border} "
            f"(toxicity probability {payload['probability'] * 100:.1f}%, "
            f"decision cutoff {payload['cutoff'] * 100:.0f}%)."
            + _domain_note(payload.get("domain"))
        )

    label = payload["compound"]
    endpoints = payload["endpoints"]
    hits = payload["flagged"]
    lines = [f"Toxicity profile for '{label}', {payload['n_endpoints']} endpoints screened:"]

    if hits:
        lines.append(f"FLAGGED ({len(hits)} endpoint(s) above their cutoff):")
        for e in hits:
            lines.append(
                f"  • {e['id']}, {e['description']}: {e['probability'] * 100:.0f}% "
                f"(cutoff {e['cutoff'] * 100:.0f}%)"
            )
    else:
        lines.append("No endpoints flagged above their cutoffs.")

    clear = len(endpoints) - len(hits)
    if clear:
        lines.append(f"{clear} other endpoint(s) below cutoff, NOT flagged by these assays "
                     f"(not proof of safety).")

    top = payload["highest_probability"]
    lines.append(
        f"Highest absolute probability: {top['id']} ({top['description']}) "
        f"at {top['probability'] * 100:.0f}%."
    )
    return "\n".join(lines) + _domain_note(payload.get("domain"))
