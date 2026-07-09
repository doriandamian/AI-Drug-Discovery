import json

import pytest

from tools.smiles_validator import validate_smiles, format_validation
from tools.property_calculator import calculate_properties, format_properties
from tools.smiles_resolver import resolve_smiles

from tests.conftest import ASPIRIN_SMILES, CAFFEINE_SMILES


def _call(tool, value):
    return tool.invoke({"compound_name": value})


def test_validate_smiles_accepts_valid_structure():
    payload = json.loads(_call(validate_smiles, ASPIRIN_SMILES))
    assert payload["status"] == "ok" and payload["valid"] is True
    assert payload["atoms"] > 0 and payload["molecular_weight"] > 0
    rendered = format_validation(payload)
    assert "Valid" in rendered and "MW:" in rendered and "Atoms:" in rendered


def test_validate_smiles_rejects_garbage():
    payload = json.loads(_call(validate_smiles, "totally-not-a-smiles"))
    assert payload["valid"] is False
    assert "Invalid input" in format_validation(payload)


def test_calculate_properties_reports_lipinski_for_aspirin():
    payload = json.loads(_call(calculate_properties, ASPIRIN_SMILES))
    assert payload["status"] == "ok"
    assert payload["lipinski_pass"] is True and payload["lipinski_violations"] == 0
    assert payload["qed"] > 0 and payload["molecular_weight"] > 0
    rendered = format_properties(payload)
    assert "Lipinski Rule of Five: PASS" in rendered
    assert "QED Score" in rendered and "Molecular Weight" in rendered


def test_calculate_properties_rejects_garbage():
    payload = json.loads(_call(calculate_properties, "not-a-molecule"))
    assert payload["status"] == "invalid"
    assert "Invalid input" in format_properties(payload)


def test_caffeine_molecular_weight_is_correct():
    payload = json.loads(_call(calculate_properties, CAFFEINE_SMILES))
    assert payload["molecular_weight"] == pytest.approx(194.19, abs=0.05)


def test_resolve_smiles_parses_valid_offline():
    smiles, mol = resolve_smiles(ASPIRIN_SMILES)
    assert mol is not None
    assert smiles == ASPIRIN_SMILES


def test_resolve_smiles_strips_smiles_tags():
    smiles, mol = resolve_smiles(f"<smiles>{ASPIRIN_SMILES}</smiles>")
    assert mol is not None
    assert smiles == ASPIRIN_SMILES


def test_resolve_smiles_unknown_returns_none_when_offline():
    smiles, mol = resolve_smiles("zzz_not_a_real_compound_zzz")
    assert smiles is None
    assert mol is None


class _FakeResp:
    def __init__(self, ok, payload=None):
        self._ok, self._payload = ok, payload or {}

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("404 PUGREST.NotFound")

    def json(self):
        return self._payload


def test_pubchem_caches_failed_lookups(monkeypatch):
    from tools import pubchem_api
    pubchem_api._CACHE.clear()
    calls = {"n": 0}

    def _fake_get(url, timeout=10):
        calls["n"] += 1
        return _FakeResp(ok=False)

    monkeypatch.setattr(pubchem_api._session, "get", _fake_get)

    a = pubchem_api.fetch_pubchem_properties.invoke({"compound_name": "zzz_unknown"})
    b = pubchem_api.fetch_pubchem_properties.invoke({"compound_name": "  ZZZ_Unknown "})
    assert calls["n"] == 1, "the failed lookup must be cached, not re-fetched"
    assert json.loads(a)["status"] == "error"
    assert a == b
    pubchem_api._CACHE.clear()


def test_pubchem_caches_successful_lookups(monkeypatch):
    from tools import pubchem_api
    pubchem_api._CACHE.clear()
    monkeypatch.setattr(pubchem_api, "upsert_compound", lambda name, props: None)
    calls = {"n": 0}
    body = {"PropertyTable": {"Properties": [
        {"IsomericSMILES": "CCO", "MolecularWeight": "46.07", "XLogP": -0.1}]}}

    def _fake_get(url, timeout=10):
        calls["n"] += 1
        return _FakeResp(ok=True, payload=body)

    monkeypatch.setattr(pubchem_api._session, "get", _fake_get)

    a = pubchem_api.fetch_pubchem_properties.invoke({"compound_name": "Ethanol"})
    b = pubchem_api.fetch_pubchem_properties.invoke({"compound_name": "ethanol"})
    assert calls["n"] == 1, "a cached success must not re-hit the network"
    assert json.loads(a)["status"] == "ok"
    assert json.loads(a)["molecular_weight"] == "46.07"
    assert a == b
    pubchem_api._CACHE.clear()
