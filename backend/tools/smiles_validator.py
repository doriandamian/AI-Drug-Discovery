import json

from langchain_core.tools import tool
from rdkit.Chem import Descriptors
from tools.smiles_resolver import resolve_smiles

__all__ = ["validate_smiles", "format_validation"]


@tool(description="""Validates a compound using RDKit and returns its basic properties if valid.

INPUT (compound_name): Either (a) a compound name already looked up with fetch_pubchem_properties (e.g. 'Aspirin'), the stored SMILES is resolved automatically, OR (b) a raw SMILES string directly (e.g. 'CC(=O)Nc1ccc(O)cc1'), it is parsed and validated as-is.

OUTPUT: a structured JSON document ({status, valid, atoms, molecular_weight}). Read values by KEY.
Call this tool BEFORE predict_toxicity whenever the SMILES was generated or modified (not taken directly from fetch_pubchem_properties).""")
def validate_smiles(compound_name: str) -> str:
    resolved_smiles, mol = resolve_smiles(compound_name)
    if mol is None:
        return json.dumps({
            "status": "invalid",
            "compound": compound_name,
            "valid": False,
            "message": (
                f"Invalid input: '{compound_name}'. Could not be parsed as a SMILES string and was "
                f"not found as a known compound name. Check for typos, or call "
                f"fetch_pubchem_properties first."
            ),
        })

    label = compound_name if (compound_name and compound_name != resolved_smiles) else resolved_smiles
    return json.dumps({
        "status": "ok",
        "compound": label,
        "valid": True,
        "atoms": mol.GetNumAtoms(),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
    })


def format_validation(payload: dict) -> str:
    if not payload.get("valid"):
        return payload.get("message", "Invalid input.")
    return (
        f"Valid: '{payload['compound']}' | Atoms: {payload['atoms']} | "
        f"MW: {payload['molecular_weight']} g/mol"
    )
