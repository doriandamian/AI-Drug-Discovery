from __future__ import annotations

import logging
import os
import random
import statistics
from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, QED

from ml.features import fingerprint_bits
from tools.toxicity_predictor import (
    get_ad_threshold,
    predict_endpoint_probs,
    safety_oracle_available,
)

logger = logging.getLogger(__name__)
RDLogger.logger().setLevel(RDLogger.CRITICAL)

W_QED, W_TOX, W_SA = 0.40, 0.40, 0.20
AD_THRESHOLD = get_ad_threshold()
_SAFETY_AVAILABLE = safety_oracle_available()
ALERT_PENALTY = 0.6
ANALOG_SIMILARITY_FLOOR = 0.30

_LIBRARY_DRUGS = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "CC(=O)Nc1ccc(O)cc1",
    "OC(=O)c1ccccc1",
    "c1ccc(cc1)S(=O)(=O)N",
    "C1CCNCC1",
    "c1ccncc1",
]


_sascorer = None
_sascorer_loaded = False
_alert_catalog = None
_alert_catalog_loaded = False


def _get_sascorer():
    global _sascorer, _sascorer_loaded
    if not _sascorer_loaded:
        _sascorer_loaded = True
        try:
            import sys
            from rdkit.Chem import RDConfig
            sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
            if sa_dir not in sys.path:
                sys.path.append(sa_dir)
            import sascorer  # type: ignore
            _sascorer = sascorer
        except Exception:
            logger.warning("SA scorer unavailable; using neutral SA fallback", exc_info=True)
    return _sascorer


def _get_alert_catalog():
    global _alert_catalog, _alert_catalog_loaded
    if not _alert_catalog_loaded:
        _alert_catalog_loaded = True
        try:
            from rdkit.Chem import FilterCatalog
            from rdkit.Chem.FilterCatalog import FilterCatalogParams
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            _alert_catalog = FilterCatalog.FilterCatalog(params)
        except Exception:
            logger.warning("Structural-alert catalog unavailable", exc_info=True)
    return _alert_catalog


def _sa_score(mol) -> float:
    sc = _get_sascorer()
    if sc is None:
        return 5.0
    try:
        return float(sc.calculateScore(mol))
    except Exception:
        return 5.0


def _count_alerts(mol) -> int:
    cat = _get_alert_catalog()
    if cat is None:
        return 0
    try:
        return len(cat.GetMatches(mol))
    except Exception:
        return 0


def _active_weights(weights: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    w_qed, w_tox, w_sa = weights if weights is not None else (W_QED, W_TOX, W_SA)
    if _SAFETY_AVAILABLE:
        return w_qed, w_tox, w_sa
    total = w_qed + w_sa
    return w_qed / total, 0.0, w_sa / total


@dataclass
class Candidate:
    smiles: str
    fitness: float
    qed: float
    tox_mean: float | None
    tox_max: float | None
    sa_score: float
    n_alerts: int
    ad_similarity: float | None
    similarity_to_seed: float | None = None


def score(smiles: str, weights: tuple[float, float, float] | None = None) -> Candidate | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    qed = float(QED.qed(mol))
    sa = _sa_score(mol)
    n_alerts = _count_alerts(mol)

    w_qed, w_tox, w_sa = _active_weights(weights)
    tox_mean = tox_max = ad = None
    if w_tox > 0.0:
        tox = predict_endpoint_probs(smiles)
        if tox is None:
            tox_term = 0.0
            logger.warning("Safety oracle returned no probabilities for a parseable "
                           "molecule; scoring its safety as worst-case: %s", smiles)
        else:
            ps = list(tox["probs"].values())
            tox_mean = statistics.mean(ps) if ps else None
            tox_max = max(ps) if ps else None
            ad = tox["ad_similarity"]
            tox_term = (1.0 - tox_mean) if tox_mean is not None else 0.0
    else:
        tox_term = 0.0

    sa_norm = max(0.0, min(1.0, (10.0 - sa) / 9.0))
    base = w_qed * qed + w_tox * tox_term + w_sa * sa_norm

    alert_factor = ALERT_PENALTY ** n_alerts
    ad_factor = 1.0
    if ad is not None and ad < AD_THRESHOLD:
        ad_factor = ad / AD_THRESHOLD

    fitness = base * alert_factor * ad_factor
    return Candidate(
        smiles=smiles, fitness=round(fitness, 4), qed=round(qed, 3),
        tox_mean=round(tox_mean, 3) if tox_mean is not None else None,
        tox_max=round(tox_max, 3) if tox_max is not None else None,
        sa_score=round(sa, 2), n_alerts=n_alerts,
        ad_similarity=round(ad, 3) if ad is not None else None,
    )


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _fragments(mol) -> set[str]:
    try:
        return set(BRICS.BRICSDecompose(mol))
    except Exception:
        return set()


def _library_fragments() -> set[str]:
    frags: set[str] = set()
    for smi in _LIBRARY_DRUGS:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            frags |= _fragments(m)
    return frags


def _library_canonicals() -> set[str]:
    out: set[str] = set()
    for smi in _LIBRARY_DRUGS:
        c = _canonical(smi)
        if c is not None:
            out.add(c)
    return out


def _tanimoto(fp_a, fp_b) -> float:
    inter = int(np.bitwise_and(fp_a, fp_b).sum())
    if inter == 0:
        return 0.0
    union = int(np.bitwise_or(fp_a, fp_b).sum())
    return inter / union if union else 0.0


def _seed_similarity(smiles: str, seed_fp) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _tanimoto(fingerprint_bits(mol), seed_fp)


def _recombine(frag_smiles, limit: int, rng: random.Random) -> list[str]:
    frags = [m for m in (Chem.MolFromSmiles(s) for s in frag_smiles) if m is not None]
    if len(frags) < 2:
        return []
    out: list[str] = []
    seen_local: set[str] = set()
    state = random.getstate()
    random.seed(rng.randint(0, 2**31 - 1))
    try:
        for i, m in enumerate(BRICS.BRICSBuild(frags, onlyCompleteMols=True)):
            if len(out) >= limit or i >= limit * 10:
                break
            try:
                Chem.SanitizeMol(m)
            except Exception:
                continue
            smi = Chem.MolToSmiles(m)
            if smi and smi not in seen_local:
                seen_local.add(smi)
                out.append(smi)
    except Exception:
        logger.warning("BRICSBuild failed", exc_info=True)
    finally:
        random.setstate(state)
    return out


def _crossover(parent_a: str, parent_b: str, rng: random.Random) -> str | None:
    ma, mb = Chem.MolFromSmiles(parent_a), Chem.MolFromSmiles(parent_b)
    if ma is None or mb is None:
        return None
    pool = _fragments(ma) | _fragments(mb)
    children = _recombine(pool, limit=1, rng=rng)
    return children[0] if children else None


def _mutate(smiles: str, fragment_vocab: list[str], rng: random.Random) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not fragment_vocab:
        return None
    frags = list(_fragments(mol))
    if len(frags) >= 2:
        frags.pop(rng.randrange(len(frags)))
    frags.append(rng.choice(fragment_vocab))
    children = _recombine(frags, limit=1, rng=rng)
    return children[0] if children else None


def _tournament(population: list[Candidate], k: int, rng: random.Random) -> Candidate:
    contenders = rng.sample(population, min(k, len(population)))
    return max(contenders, key=lambda c: c.fitness)


def design_molecules(
    seed_smiles: str,
    n_generations: int = 6,
    population_size: int = 30,
    top_k: int = 5,
    elite_size: int = 4,
    tournament_size: int = 3,
    mutation_rate: float = 0.3,
    eval_budget: int = 200,
    analog_similarity_floor: float = ANALOG_SIMILARITY_FLOOR,
    rng_seed: int = 0,
    weights: tuple[float, float, float] | None = None,
) -> dict:
    rng = random.Random(rng_seed)

    seed_canon = _canonical(seed_smiles)
    if seed_canon is None:
        return {"seed": None, "candidates": [], "generations_best": [],
                "stats": {"note": "seed SMILES could not be parsed",
                          "safety_optimized": _SAFETY_AVAILABLE}}

    seed_mol = Chem.MolFromSmiles(seed_canon)
    seed_fp = fingerprint_bits(seed_mol)
    seed_cand = score(seed_canon, weights)
    if seed_cand is not None:
        seed_cand.similarity_to_seed = 1.0

    exclude = {seed_canon} | _library_canonicals()
    fragment_vocab = list(_fragments(seed_mol) | _library_fragments())

    scored: dict[str, Candidate] = {}
    seen: set[str] = {seed_canon}

    def consider(smi: str | None) -> Candidate | None:
        if smi is None:
            return None
        canon = _canonical(smi)
        if canon is None or canon in seen:
            return None
        seen.add(canon)
        if canon in exclude:
            return None
        sim = _seed_similarity(canon, seed_fp)
        if sim is None or sim < analog_similarity_floor:
            return None
        if len(scored) >= eval_budget:
            return None
        cand = score(canon, weights)
        if cand is None:
            return None
        cand.similarity_to_seed = round(sim, 3)
        scored[canon] = cand
        return cand

    population: list[Candidate] = []
    for smi in _recombine(fragment_vocab, limit=population_size * 3, rng=rng):
        cand = consider(smi)
        if cand is not None:
            population.append(cand)
        if len(population) >= population_size:
            break

    if not population:
        return {
            "seed": asdict(seed_cand) if seed_cand else None,
            "candidates": [], "generations_best": [],
            "stats": {
                "generated": len(seen) - 1, "scored": 0, "generations": 0,
                "safety_optimized": _SAFETY_AVAILABLE,
                "analog_similarity_floor": analog_similarity_floor,
                "note": (f"no analogs above the similarity floor "
                         f"{analog_similarity_floor:.2f} could be generated; the "
                         f"seed may not decompose into recombinable BRICS fragments"),
            },
        }

    generations_best: list[float] = []
    for _gen in range(n_generations):
        generations_best.append(round(max(c.fitness for c in scored.values()), 4))
        if len(scored) >= eval_budget:
            break

        population.sort(key=lambda c: c.fitness, reverse=True)
        elites = population[:elite_size]
        target_offspring = max(0, population_size - len(elites))
        offspring: list[Candidate] = []
        attempts = 0
        max_attempts = population_size * 8
        while (len(offspring) < target_offspring and attempts < max_attempts
               and len(scored) < eval_budget):
            attempts += 1
            parent_a = _tournament(population, tournament_size, rng)
            parent_b = _tournament(population, tournament_size, rng)
            child = _crossover(parent_a.smiles, parent_b.smiles, rng)
            if child is None:
                child = _mutate(parent_a.smiles, fragment_vocab, rng)
            elif rng.random() < mutation_rate:
                child = _mutate(child, fragment_vocab, rng) or child
            cand = consider(child)
            if cand is not None:
                offspring.append(cand)

        population = elites + offspring
        if not population:
            break

    ranked = sorted(scored.values(), key=lambda c: c.fitness, reverse=True)[:top_k]

    stats = {
        "generated": len(seen) - 1,
        "scored": len(scored),
        "generations": len(generations_best),
        "safety_optimized": _SAFETY_AVAILABLE,
        "analog_similarity_floor": analog_similarity_floor,
    }
    if not _SAFETY_AVAILABLE:
        stats["note"] = (
            "SAFETY NOT OPTIMIZED: no multitask toxicity model is loaded, so the "
            "fitness reflects only drug-likeness and synthesizability. Candidates "
            "were NOT screened for predicted toxicity."
        )

    return {
        "seed": asdict(seed_cand) if seed_cand else None,
        "candidates": [asdict(c) for c in ranked],
        "generations_best": generations_best,
        "stats": stats,
    }
