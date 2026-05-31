import os
import numpy as np
import joblib
from langchain_core.tools import tool
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "toxicity_model.pkl")

_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    """Converts a SMILES string to a Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = mfpgen.GetFingerprintAsNumPy(mol)
    
    return fp.reshape(1, -1)

@tool(description="""Predicts if a chemical molecule is toxic using Machine Learning.
CRITICAL RULES:
1. FOR KNOWN DRUGS: You MUST call fetch_pubchem_properties FIRST to extract the official SMILES. Do not guess it.
2. FOR NEW INVENTIONS: You must generate the SMILES yourself. Be very careful with chemical rules (e.g., use 'c1ccccc1' for aromatic benzene rings to avoid valency errors).
3. IF YOU GET AN ERROR: If this tool returns 'Error: Invalid SMILES string', it means you made a chemistry syntax mistake. You MUST correct the SMILES string and CALL THIS TOOL AGAIN immediately. Do not just apologize in text.""")
def predict_toxicity(smiles: str) -> str:
    if _model is None:
        return "Error: Toxicity prediction model not found. Please train the model first."

    smiles = smiles.replace("<smiles>", "").replace("</smiles>", "").strip()

    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return f"Error: Invalid SMILES string provided: '{smiles}'"

    try:
        prediction = _model.predict(fp)[0]
        probability = _model.predict_proba(fp)[0]

        if prediction == 1:
            risk = probability[1] * 100
            return f"Prediction: Toxic (Risk Score: {risk:.2f}%) for SMILES: '{smiles}'"
        else:
            safety = probability[0] * 100
            return f"Prediction: Safe (Safety Score: {safety:.2f}%) for SMILES: '{smiles}'"
    except Exception as e:
        return f"Error during prediction: {str(e)}"