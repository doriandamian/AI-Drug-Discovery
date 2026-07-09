import contextvars
import json
import re

from langchain_core.tools import tool

from tools.smiles_resolver import resolve_smiles
from ml.generative import design_molecules

__all__ = ["detect_design_goal", "set_design_goal", "design_analogs", "format_design_report"]

DESIGN_GOALS: dict[str, tuple[float, float, float]] = {
    "safer":     (0.25, 0.60, 0.15),
    "drug_like": (0.60, 0.25, 0.15),
    "balanced":  (0.40, 0.40, 0.20),
}
_GOAL_LABEL = {
    "safer": "predicted safety (lower toxicity weighted highest)",
    "drug_like": "drug-likeness / QED (weighted highest)",
    "balanced": "a balance of drug-likeness, predicted safety, and synthesizability",
}

_SAFER_RE = re.compile(
    r"safer|less\s+toxic|lower\s+toxic|reduc\w*\s+toxic|non-?toxic|detoxif", re.IGNORECASE
)
_DRUGLIKE_RE = re.compile(
    r"more\s+drug-?like|drug-?likeness"
    r"|(?:improv\w*|better|higher|maximi[sz]\w*|increas\w*)\s+(?:the\s+)?qed",
    re.IGNORECASE,
)


def detect_design_goal(task: str) -> str:
    t = task or ""
    if _SAFER_RE.search(t):
        return "safer"
    if _DRUGLIKE_RE.search(t):
        return "drug_like"
    return "balanced"


_active_goal: contextvars.ContextVar[str] = contextvars.ContextVar(
    "design_goal", default="balanced"
)


def set_design_goal(goal: str) -> None:
    _active_goal.set(goal if goal in DESIGN_GOALS else "balanced")

_METRICS = (
    ("fitness", "fitness", True),
    ("qed", "QED", True),
    ("tox", "tox", False),
    ("sa", "SA", False),
)
_METRIC_LABEL = {key: label for key, label, _ in _METRICS}


def _node(c: dict) -> dict:
    return {
        "smiles": c["smiles"],
        "fitness": round(c["fitness"], 2),
        "qed": round(c["qed"], 2),
        "tox": None if c["tox_mean"] is None else round(c["tox_mean"], 2),
        "sa": round(c["sa_score"], 2),
        "alerts": c["n_alerts"],
        "ad_similarity": None if c["ad_similarity"] is None else round(c["ad_similarity"], 2),
        "similarity_to_seed": (
            None if c.get("similarity_to_seed") is None else round(c["similarity_to_seed"], 2)
        ),
    }


def _vs_seed(cand: dict, seed: dict) -> dict:
    out: dict[str, dict] = {}
    for key, _label, higher_better in _METRICS:
        cv, sv = cand.get(key), seed.get(key)
        if cv is None or sv is None:
            continue
        delta = round(cv - sv, 2)
        if delta == 0:
            direction = "unchanged"
        else:
            improved = (delta > 0) if higher_better else (delta < 0)
            direction = "improved" if improved else "worsened"
        out[key] = {"delta": delta, "direction": direction}
    return out


_CAVEATS = [
    "These are algorithmically generated hypotheses, NOT real or validated drugs.",
    "Toxicity is a prediction from an in-vitro-trained model (Tox21/ClinTox); it does "
    "not establish safety. A low applicability-domain similarity (AD) means the "
    "prediction is a weak hint.",
    "SA is a heuristic synthesizability estimate; fitness is a triage signal only. "
    "Any candidate must be reviewed and tested experimentally.",
]


@tool(description="""Generates and ranks NOVEL candidate analogs of a known compound, the molecular DESIGN / optimization tool.

WHEN TO USE: the user asks to design, generate, propose, optimize, "improve", or make a "more drug-like" or "safer / less toxic" version of a molecule. This is the ONLY way to create structures; never invent a SMILES by hand.

INPUT (compound_name): the seed compound's NAME (e.g. 'Aspirin'). Never pass a SMILES string.

HOW IT WORKS: RDKit recombines the seed's BRICS fragments over a few generations and scores every candidate with a multi-objective oracle, drug-likeness (QED), the local toxicity model (lower is better), synthetic accessibility (SA), minus penalties for PAINS/Brenk structural alerts and for drifting outside the toxicity model's applicability domain.

OUTPUT: a STRUCTURED JSON document with named fields, the seed baseline and the top-ranked candidates, each with its SMILES, labeled scores, and a per-metric comparison to the seed. Read every value by its KEY; never re-parse the numbers positionally. The candidates are UNVALIDATED computational proposals, report them as design hypotheses for a chemist to evaluate, never as established drugs.""")
def design_analogs(compound_name: str) -> str:
    seed_smiles, mol = resolve_smiles(compound_name)
    if mol is None:
        return json.dumps({
            "status": "unresolved",
            "compound": compound_name,
            "message": (
                f"Could not resolve '{compound_name}' to a structure. Call "
                f"fetch_pubchem_properties first, then retry by name."
            ),
        })

    goal = _active_goal.get()
    weights = DESIGN_GOALS[goal]

    result = design_molecules(seed_smiles, top_k=3, weights=weights)
    cands_raw = result["candidates"]
    stats = result["stats"]

    if not cands_raw:
        note = stats.get("note", "no valid analogs could be generated")
        return json.dumps({
            "status": "no_candidates",
            "compound": compound_name,
            "message": (
                f"Design run for '{compound_name}' produced no candidates ({note}). "
                f"The seed may not decompose into recombinable BRICS fragments."
            ),
        })

    seed = _node(result["seed"])
    candidates = []
    for i, c in enumerate(cands_raw, 1):
        node = _node(c)
        node["rank"] = i
        node["vs_seed"] = _vs_seed(node, seed)
        candidates.append(node)

    payload = {
        "status": "ok",
        "compound": compound_name,
        "goal": goal,
        "goal_label": _GOAL_LABEL[goal],
        "safety_optimized": stats.get("safety_optimized", False),
        "n_scored": stats.get("scored", len(candidates)),
        "seed": seed,
        "candidates": candidates,
        "generations_best": result.get("generations_best", []),
        "caveats": _CAVEATS,
    }
    return json.dumps(payload)

def _fmt_scores(node: dict) -> str:
    tox = "n/a" if node["tox"] is None else f"{node['tox']:.2f}"
    ad = "n/a" if node["ad_similarity"] is None else f"{node['ad_similarity']:.2f}"
    sim = node.get("similarity_to_seed")
    sim_str = "" if sim is None else f" seed-sim={sim:.2f}"
    return (
        f"fitness={node['fitness']:.2f} QED={node['qed']:.2f} "
        f"tox={tox} SA={node['sa']:.2f} alerts={node['alerts']} AD={ad}{sim_str}"
    )


def _fmt_comparison(vs_seed: dict) -> str:
    parts = []
    for key, _label, _ in _METRICS:
        d = vs_seed.get(key)
        if d is None:
            continue
        parts.append(f"{_METRIC_LABEL[key]} {d['direction']} ({d['delta']:+.2f})")
    return ", ".join(parts)


def format_design_report(payload: dict) -> str:
    if payload.get("status") != "ok":
        return payload.get("message", "Design run failed.")

    seed = payload["seed"]
    lines = [
        f"Designed {payload['n_scored']} analog(s) of '{payload['compound']}' "
        f"(UNVALIDATED computational proposals).",
    ]
    goal_label = payload.get("goal_label")
    if goal_label:
        lines.append(f"Optimized for: {goal_label}.")
    lines += [
        "",
        "SEED BASELINE:",
        f"  <smiles>{seed['smiles']}</smiles>",
        f"  {_fmt_scores(seed)}",
        "",
        "TOP CANDIDATES (scores are labeled, read each by its label, never by position):",
    ]
    for c in payload["candidates"]:
        lines.append(f"  {c['rank']}. <smiles>{c['smiles']}</smiles>")
        lines.append(f"     {_fmt_scores(c)}")
        comparison = _fmt_comparison(c.get("vs_seed", {}))
        if comparison:
            lines.append(f"     vs seed: {comparison}")

    if payload.get("generations_best"):
        lines += [
            "",
            f"Optimization curve (best fitness per generation): {payload['generations_best']}",
        ]

    if payload.get("safety_optimized") is False:
        lines += [
            "",
            "⚠ SAFETY WAS NOT OPTIMIZED: the toxicity model was unavailable, so this "
            "ranking reflects only drug-likeness and synthesizability, NOT predicted "
            "toxicity. Do not read the fitness as a safety signal.",
        ]

    caveats = payload.get("caveats", [])
    if caveats:
        lines.append("")
        lines.append("CAVEATS: state these to the user:")
        lines += [f"- {c}" for c in caveats]

    return "\n".join(lines)
