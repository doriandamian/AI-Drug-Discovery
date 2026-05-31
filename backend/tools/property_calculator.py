from langchain_core.tools import tool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED


@tool(description="""Calculates drug-likeness and ADMET-related molecular properties from a SMILES string using RDKit.
Use this tool when the user asks about drug-likeness, Lipinski's Rule of Five, QED score, or detailed molecular properties.
Also useful after modifying a molecule to evaluate whether the new structure is suitable as a drug candidate.""")
def calculate_properties(smiles: str) -> str:
    smiles = smiles.replace("<smiles>", "").replace("</smiles>", "").strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"Invalid SMILES: '{smiles}'. Cannot calculate properties."

    mw        = round(Descriptors.MolWt(mol), 2)
    logp      = round(Descriptors.MolLogP(mol), 2)
    hbd       = rdMolDescriptors.CalcNumHBD(mol)
    hba       = rdMolDescriptors.CalcNumHBA(mol)
    tpsa      = round(Descriptors.TPSA(mol), 2)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    qed_score = round(QED.qed(mol), 3)
    rings     = rdMolDescriptors.CalcNumRings(mol)

    lipinski_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])

    lipinski_pass = "PASS" if lipinski_violations == 0 else f"FAIL ({lipinski_violations} violation{'s' if lipinski_violations > 1 else ''})"

    return (
        f"Molecular Properties for '{smiles}':\n"
        f"- Molecular Weight:       {mw} g/mol  (limit: ≤500)\n"
        f"- LogP (lipophilicity):   {logp}       (limit: ≤5)\n"
        f"- H-Bond Donors (HBD):   {hbd}         (limit: ≤5)\n"
        f"- H-Bond Acceptors (HBA):{hba}         (limit: ≤10)\n"
        f"- TPSA:                   {tpsa} Å²    (oral: <140)\n"
        f"- Rotatable Bonds:        {rot_bonds}  (oral: <10)\n"
        f"- Rings:                  {rings}\n"
        f"- QED Score:              {qed_score}  (0=low, 1=ideal drug-likeness)\n"
        f"- Lipinski Rule of Five:  {lipinski_pass}"
    )
