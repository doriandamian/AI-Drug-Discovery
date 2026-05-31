from langchain_core.tools import tool
from rdkit import Chem
from rdkit.Chem import Descriptors


@tool(description="""Validates a SMILES string using RDKit and returns its basic properties if valid.
Call this tool BEFORE predict_toxicity whenever the SMILES was generated or modified (not taken directly from fetch_pubchem_properties).
Do NOT call it for SMILES that came directly from fetch_pubchem_properties — those are already trusted.""")
def validate_smiles(smiles: str) -> str:
    smiles = smiles.replace("<smiles>", "").replace("</smiles>", "").strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"Invalid SMILES: '{smiles}'. The string could not be parsed. Check for mismatched parentheses, invalid atom symbols, or broken ring closures."

    mw = round(Descriptors.MolWt(mol), 2)
    atoms = mol.GetNumAtoms()
    return f"Valid SMILES: '{smiles}' | Atoms: {atoms} | MW: {mw} g/mol"
