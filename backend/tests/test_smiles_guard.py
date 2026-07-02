import pytest

from agents import smiles_guard

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


@pytest.fixture(autouse=True)
def fresh_window():
    smiles_guard.reset()
    yield
    smiles_guard.reset()


def test_strips_ungrounded_smiles():
    answer = f"- **Aspirin**: <smiles>{ASPIRIN}</smiles>\n- **Ibuprofen**: <smiles>{IBUPROFEN}</smiles>"
    clean, removed = smiles_guard.sanitize(answer)
    assert "<smiles>" not in clean
    assert smiles_guard.WITHHELD_MARKER in clean
    assert len(removed) == 2


def test_keeps_smiles_a_tool_returned():
    smiles_guard.record_from_text(f"1 row(s):\n  • name=Aspirin, smiles={ASPIRIN}")
    clean, removed = smiles_guard.sanitize(f"The structure is <smiles>{ASPIRIN}</smiles>.")
    assert removed == []
    assert f"<smiles>{ASPIRIN}</smiles>" in clean


def test_keeps_user_provided_smiles():
    smiles_guard.record_user_message(f"Is this drug-like? {ASPIRIN}")
    clean, removed = smiles_guard.sanitize(f"You gave <smiles>{ASPIRIN}</smiles>.")
    assert removed == []
    assert "<smiles>" in clean


def test_matches_across_reformatting():
    smiles_guard.record_from_text("smiles=OC(=O)c1ccccc1OC(C)=O")  # aspirin, other order
    clean, removed = smiles_guard.sanitize(f"<smiles>{ASPIRIN}</smiles>")
    assert removed == []


def test_strips_invalid_smiles():
    clean, removed = smiles_guard.sanitize("<smiles>this-is-not-a-molecule</smiles>")
    assert removed == ["this-is-not-a-molecule"]
    assert smiles_guard.WITHHELD_MARKER in clean


def test_does_not_record_compound_names_as_smiles():
    smiles_guard.record_from_text("Valid: 'Aspirin' | Atoms: 13 | MW: 180.16 g/mol")
    _, removed = smiles_guard.sanitize(f"<smiles>{ASPIRIN}</smiles>")
    assert len(removed) == 1
