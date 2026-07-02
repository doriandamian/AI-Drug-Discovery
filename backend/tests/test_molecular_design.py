import json

import pytest
from rdkit import Chem

from ml import generative
from agents import smiles_guard
from tools import toxicity_predictor
from tools.molecular_design import (
    design_analogs, format_design_report, detect_design_goal, set_design_goal,
    DESIGN_GOALS,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_CANON = Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN))


def test_score_returns_sane_metrics():
    c = generative.score(ASPIRIN)
    assert c is not None
    assert 0.0 <= c.qed <= 1.0
    assert 0.0 <= c.fitness <= 1.0
    assert c.sa_score > 0
    assert c.tox_mean is None or 0.0 <= c.tox_mean <= 1.0
    assert isinstance(c.n_alerts, int) and c.n_alerts >= 0


def test_score_rejects_invalid_smiles():
    assert generative.score("not-a-molecule") is None


def test_goal_weights_change_the_fitness(monkeypatch):
    assert generative._SAFETY_AVAILABLE, "test assumes the multitask model is loaded"
    monkeypatch.setattr(
        generative, "predict_endpoint_probs",
        lambda smi: {"probs": {"e1": 0.9, "e2": 0.9}, "ad_similarity": 1.0},
    )
    safer = generative.score(ASPIRIN, weights=(0.25, 0.60, 0.15))
    drug_like = generative.score(ASPIRIN, weights=(0.60, 0.25, 0.15))
    assert safer.fitness < drug_like.fitness


def test_ad_threshold_is_sourced_from_the_model_bundle():
    assert generative.AD_THRESHOLD == toxicity_predictor.get_ad_threshold()


def test_missing_safety_score_is_penalized_not_neutralized(monkeypatch):
    assert generative._SAFETY_AVAILABLE, "test assumes the multitask model is loaded"

    monkeypatch.setattr(generative, "predict_endpoint_probs", lambda smi: None)
    cand_none = generative.score(ASPIRIN)

    monkeypatch.setattr(
        generative, "predict_endpoint_probs",
        lambda smi: {"probs": {"e1": 0.0, "e2": 0.0}, "ad_similarity": 1.0},
    )
    cand_safe = generative.score(ASPIRIN)

    assert cand_none.tox_mean is None, "a missing safety score must not be reported as a value"
    assert cand_none.fitness < cand_safe.fitness, "missing safety must rank DOWN, not neutral"


def test_safety_weights_renormalize_when_oracle_unavailable(monkeypatch):
    monkeypatch.setattr(generative, "_SAFETY_AVAILABLE", False)

    def explode(smi):
        raise AssertionError("the safety oracle must NOT be consulted when unavailable")

    monkeypatch.setattr(generative, "predict_endpoint_probs", explode)

    w_qed, w_tox, w_sa = generative._active_weights()
    assert w_tox == 0.0
    assert w_qed + w_sa == pytest.approx(1.0)

    c = generative.score(ASPIRIN)
    assert c.tox_mean is None and c.tox_max is None


def test_design_reports_safety_optimized_flag():
    run = generative.design_molecules(ASPIRIN, n_generations=1, population_size=8,
                                      eval_budget=20, rng_seed=0)
    assert run["stats"]["safety_optimized"] is generative._SAFETY_AVAILABLE


@pytest.fixture(scope="module")
def small_run():
    return generative.design_molecules(
        ASPIRIN, n_generations=3, population_size=14, eval_budget=50, rng_seed=0
    )


def test_design_produces_valid_distinct_candidates(small_run):
    cands = small_run["candidates"]
    assert cands, "expected at least one candidate"
    seen = set()
    for c in cands:
        mol = Chem.MolFromSmiles(c["smiles"])
        assert mol is not None, f"invalid SMILES generated: {c['smiles']}"
        assert c["smiles"] != ASPIRIN_CANON, "the seed must not be returned as a candidate"
        assert c["smiles"] not in seen, "candidates must be unique"
        seen.add(c["smiles"])


def test_candidates_clear_the_analog_similarity_floor(small_run):
    floor = small_run["stats"]["analog_similarity_floor"]
    for c in small_run["candidates"]:
        assert c["similarity_to_seed"] is not None
        assert c["similarity_to_seed"] >= floor, (
            f"{c['smiles']} (seed-sim {c['similarity_to_seed']}) is below the analog floor {floor}"
        )


def test_library_drugs_are_never_returned_as_candidates(small_run):
    library = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in generative._LIBRARY_DRUGS}
    returned = {c["smiles"] for c in small_run["candidates"]}
    assert returned.isdisjoint(library), "a seeded library drug was returned as a designed analog"


def test_candidates_ranked_by_descending_fitness(small_run):
    fits = [c["fitness"] for c in small_run["candidates"]]
    assert fits == sorted(fits, reverse=True)


def test_generations_best_is_monotonic_nondecreasing(small_run):
    curve = small_run["generations_best"]
    assert all(b >= a for a, b in zip(curve, curve[1:])), curve


def test_design_is_reproducible():
    a = generative.design_molecules(ASPIRIN, n_generations=2, population_size=12, eval_budget=30, rng_seed=0)
    b = generative.design_molecules(ASPIRIN, n_generations=2, population_size=12, eval_budget=30, rng_seed=0)
    assert [c["smiles"] for c in a["candidates"]] == [c["smiles"] for c in b["candidates"]]


def test_unparseable_seed_returns_a_clear_note():
    run = generative.design_molecules("not-a-molecule")
    assert run["candidates"] == []
    assert "could not be parsed" in run["stats"]["note"]


def _fake_run(cand_smiles, *, cand_tox=0.07, safety=True):
    return {
        "seed": {"smiles": ASPIRIN, "fitness": 0.82, "qed": 0.82,
                 "tox_mean": None if not safety else 0.06,
                 "tox_max": 0.2, "sa_score": 2.19, "n_alerts": 0,
                 "ad_similarity": None if not safety else 1.0, "similarity_to_seed": 1.0},
        "candidates": [{"smiles": cand_smiles, "fitness": 0.89, "qed": 0.89,
                        "tox_mean": None if not safety else cand_tox,
                        "tox_max": 0.2, "sa_score": 1.51, "n_alerts": 0,
                        "ad_similarity": None if not safety else 0.82,
                        "similarity_to_seed": 0.55}],
        "generations_best": [0.85, 0.89],
        "stats": {"scored": 12, "safety_optimized": safety, "analog_similarity_floor": 0.30},
    }


def test_tool_returns_named_field_json(monkeypatch):
    cand_smiles = "CC(C)Cc1ccc(-c2ccc(C(=O)O)cc2)cc1"
    monkeypatch.setattr("tools.molecular_design.design_molecules",
                        lambda *a, **k: _fake_run(cand_smiles))

    payload = json.loads(design_analogs.invoke({"compound_name": ASPIRIN}))
    assert payload["status"] == "ok"
    assert payload["safety_optimized"] is True
    assert payload["seed"]["smiles"] == ASPIRIN and payload["seed"]["tox"] == 0.06
    cand = payload["candidates"][0]
    assert cand["rank"] == 1 and cand["smiles"] == cand_smiles
    assert cand["fitness"] == 0.89 and cand["qed"] == 0.89
    assert cand["tox"] == 0.07 and cand["sa"] == 1.51 and cand["ad_similarity"] == 0.82


def test_vs_seed_directions_are_computed_per_metric(monkeypatch):
    cand_smiles = "CC(C)Cc1ccc(-c2ccc(C(=O)O)cc2)cc1"
    monkeypatch.setattr("tools.molecular_design.design_molecules",
                        lambda *a, **k: _fake_run(cand_smiles, cand_tox=0.09))
    vs = json.loads(design_analogs.invoke({"compound_name": ASPIRIN}))["candidates"][0]["vs_seed"]

    assert vs["fitness"] == {"delta": 0.07, "direction": "improved"}   # higher better
    assert vs["qed"] == {"delta": 0.07, "direction": "improved"}       # higher better
    assert vs["tox"] == {"delta": 0.03, "direction": "worsened"}       # LOWER better
    assert vs["sa"] == {"delta": -0.68, "direction": "improved"}       # LOWER better


def test_unresolvable_name_is_a_status_not_an_answer():
    payload = json.loads(design_analogs.invoke({"compound_name": "zzz_not_a_real_compound"}))
    assert payload["status"] == "unresolved"
    assert "could not resolve" in payload["message"].lower()


def test_report_is_labeled_and_guard_grounded(monkeypatch):
    cand_smiles = "CC(C)Cc1ccc(-c2ccc(C(=O)O)cc2)cc1"
    monkeypatch.setattr("tools.molecular_design.design_molecules",
                        lambda *a, **k: _fake_run(cand_smiles, cand_tox=0.09))
    report = format_design_report(json.loads(design_analogs.invoke({"compound_name": ASPIRIN})))

    assert f"<smiles>{cand_smiles}</smiles>" in report
    assert "UNVALIDATED" in report and "CAVEAT" in report.upper()
    assert "tox=0.09" in report and "SA=1.51" in report and "QED=0.89" in report
    assert "tox worsened (+0.03)" in report
    assert "fitness improved (+0.07)" in report

    smiles_guard.reset()
    smiles_guard.record_from_text(report)
    clean, removed = smiles_guard.sanitize(f"Proposed: <smiles>{cand_smiles}</smiles>")
    assert removed == [] and cand_smiles in clean
    smiles_guard.reset()


def test_json_tool_output_grounds_smiles_directly(monkeypatch):
    cand_smiles = "CC(C)Cc1ccc(-c2ccc(C(=O)O)cc2)cc1"
    monkeypatch.setattr("tools.molecular_design.design_molecules",
                        lambda *a, **k: _fake_run(cand_smiles))
    raw_json = design_analogs.invoke({"compound_name": ASPIRIN})

    smiles_guard.reset()
    smiles_guard.record_from_text(raw_json)
    _, removed = smiles_guard.sanitize(f"<smiles>{cand_smiles}</smiles>")
    assert removed == []
    smiles_guard.reset()


def test_report_warns_loudly_when_safety_not_optimized(monkeypatch):
    monkeypatch.setattr("tools.molecular_design.design_molecules",
                        lambda *a, **k: _fake_run("CCO", safety=False))
    report = format_design_report(json.loads(design_analogs.invoke({"compound_name": ASPIRIN})))
    assert "SAFETY WAS NOT OPTIMIZED" in report
    assert "tox " not in report.split("vs seed:")[1] if "vs seed:" in report else True


def test_report_renders_failure_status_as_message():
    msg = format_design_report({"status": "no_candidates", "compound": "X",
                                "message": "Design run for 'X' produced no candidates."})
    assert msg == "Design run for 'X' produced no candidates."


@pytest.fixture(autouse=True)
def _reset_design_goal():
    set_design_goal("balanced")
    yield
    set_design_goal("balanced")


def test_detect_design_goal_maps_phrasing_to_profile():
    assert detect_design_goal("design a safer version of aspirin") == "safer"
    assert detect_design_goal("make ibuprofen less toxic") == "safer"
    assert detect_design_goal("a more drug-like analog of caffeine") == "drug_like"
    assert detect_design_goal("improve the QED of aspirin") == "drug_like"
    assert detect_design_goal("a safer, more drug-like aspirin") == "safer"
    assert detect_design_goal("design some analogs of aspirin") == "balanced"
    assert detect_design_goal("") == "balanced"


def test_balanced_profile_equals_the_engine_default_weights():
    assert DESIGN_GOALS["balanced"] == (generative.W_QED, generative.W_TOX, generative.W_SA)


def test_active_goal_threads_its_weight_profile_into_the_engine(monkeypatch):
    captured = {}

    def _capture(seed_smiles, *a, **k):
        captured["weights"] = k.get("weights")
        return _fake_run("CC(C)Cc1ccc(-c2ccc(C(=O)O)cc2)cc1")

    monkeypatch.setattr("tools.molecular_design.design_molecules", _capture)

    set_design_goal("safer")
    payload = json.loads(design_analogs.invoke({"compound_name": ASPIRIN}))
    assert captured["weights"] == DESIGN_GOALS["safer"]
    assert payload["goal"] == "safer"
    assert "safety" in payload["goal_label"].lower()


def test_report_states_the_optimization_goal(monkeypatch):
    monkeypatch.setattr("tools.molecular_design.design_molecules",
                        lambda *a, **k: _fake_run("CC(C)Cc1ccc(-c2ccc(C(=O)O)cc2)cc1"))
    set_design_goal("drug_like")
    report = format_design_report(json.loads(design_analogs.invoke({"compound_name": ASPIRIN})))
    assert "Optimized for:" in report and "QED" in report
