import pytest

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
ETHANOL_SMILES = "CCO"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Disable the Neo4j and PubChem fallbacks in smiles_resolver for every test."""
    monkeypatch.setattr("tools.smiles_resolver.get_compound", lambda name: None)
    monkeypatch.setattr(
        "tools.smiles_resolver._fetch_smiles_from_pubchem", lambda name: None
    )
