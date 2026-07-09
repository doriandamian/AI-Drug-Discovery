import json

from langchain_core.tools import tool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from tools.smiles_resolver import resolve_smiles

__all__ = ["calculate_properties", "format_properties"]

_LIPINSKI_LIMITS = {"mw": 500, "logp": 5, "hbd": 5, "hba": 10}

_FUNCTIONAL_GROUP_SMARTS: list[tuple[str, str]] = [
    ("amide",           "[NX3][CX3](=[OX1])"),
    ("sulfonamide",     "[#16X4](=[OX1])(=[OX1])[NX3]"),
    ("carboxylic acid", "[CX3](=O)[OX2H1]"),
    ("sulfonic acid",   "[#16X4](=[OX1])(=[OX1])[OX2H]"),
    ("ester",           "[#6][CX3](=O)[OX2H0][#6]"),
    ("ketone",          "[#6][CX3](=O)[#6]"),
    ("aldehyde",        "[CX3H1](=O)"),
    ("phenol",          "[OX2H]c"),
    ("hydroxyl",        "[OX2H][CX4]"),
    ("primary amine",   "[NX3;H2;!$(NC=O)]"),
    ("secondary amine", "[NX3;H1;!$(NC=O)]"),
    ("tertiary amine",  "[NX3;H0;!$(NC=O);!$(N=*)]"),
    ("nitrile",         "[NX1]#[CX2]"),
    ("nitro",           "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("halide",          "[F,Cl,Br,I]"),
    ("epoxide",         "[OX2r3]"),
]
_COMPILED_SMARTS: list[tuple[str, object]] = [
    (name, Chem.MolFromSmarts(smarts))
    for name, smarts in _FUNCTIONAL_GROUP_SMARTS
]


def _detect_functional_groups(mol) -> list[str]:
    groups = [name for name, patt in _COMPILED_SMARTS if patt and mol.HasSubstructMatch(patt)]
    if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
        groups.append("aromatic ring")
    return groups


@tool(description="""Calculates drug-likeness and ADMET-related molecular properties from a SMILES string using RDKit.
Use this tool when the user asks about drug-likeness, Lipinski's Rule of Five, QED score, or detailed molecular properties.

INPUT (compound_name): The compound name already looked up with fetch_pubchem_properties (e.g. 'Aspirin'). Always pass the compound name, never a SMILES string.

OUTPUT: a structured JSON document with named numeric fields (molecular_formula, molecular_weight, logp, hbd, hba, tpsa, rotatable_bonds, rings, qed, lipinski_violations, lipinski_pass). Read every value by its KEY.""")
def calculate_properties(compound_name: str) -> str:
    resolved_smiles, mol = resolve_smiles(compound_name)
    if mol is None:
        return json.dumps({
            "status": "invalid",
            "compound": compound_name,
            "message": (
                f"Invalid input: '{compound_name}' could not be parsed as a SMILES string or "
                f"found as a known compound name. Call fetch_pubchem_properties first."
            ),
        })

    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    violations = sum([
        mw > _LIPINSKI_LIMITS["mw"],
        logp > _LIPINSKI_LIMITS["logp"],
        hbd > _LIPINSKI_LIMITS["hbd"],
        hba > _LIPINSKI_LIMITS["hba"],
    ])

    label = compound_name if (compound_name and compound_name != resolved_smiles) else resolved_smiles
    return json.dumps({
        "status": "ok",
        "compound": label,
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "qed": round(QED.qed(mol), 3),
        "lipinski_violations": violations,
        "lipinski_pass": violations == 0,
        "functional_groups": _detect_functional_groups(mol),
    })


def format_properties(payload: dict) -> str:
    if payload.get("status") != "ok":
        return payload.get("message", "Property calculation failed.")

    v = payload["lipinski_violations"]
    lipinski = (
        f"PASS: 0 violations" if payload["lipinski_pass"]
        else f"FAIL: {v} violation{'s' if v != 1 else ''}"
    )
    fg = payload.get("functional_groups") or []
    fg_line = f"- Functional Groups: {', '.join(fg)}" if fg else "- Functional Groups: none detected"
    return (
        f"Molecular Properties for '{payload['compound']}':\n"
        f"- Molecular Formula: {payload['molecular_formula']}\n"
        f"- Molecular Weight: {payload['molecular_weight']} g/mol  (limit ≤500)\n"
        f"- LogP (lipophilicity): {payload['logp']}  (limit ≤5)\n"
        f"- H-Bond Donors (HBD): {payload['hbd']}  (limit ≤5)\n"
        f"- H-Bond Acceptors (HBA): {payload['hba']}  (limit ≤10)\n"
        f"- TPSA: {payload['tpsa']} Å²  (oral <140)\n"
        f"- Rotatable Bonds: {payload['rotatable_bonds']}  (oral <10)\n"
        f"- Rings: {payload['rings']}\n"
        f"- QED Score: {payload['qed']}  (0=low, 1=ideal drug-likeness)\n"
        f"- Lipinski Rule of Five: {lipinski}\n"
        f"{fg_line}"
    )
