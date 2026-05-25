import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    """Converts a SMILES string to a Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((nBits,))
    mfpgen=rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = mfpgen.GetFingerprint(mol)
    return fp

def train():
    print("Downloading ClinTox dataset...")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"

    try:
        df = pd.read_csv(url)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    df = df.dropna(subset=['smiles', 'CT_TOX'])

    print("Preparing training data...")
    X = np.array([smiles_to_fingerprint(smiles) for smiles in df['smiles']])
    y = np.array([label for label in df['CT_TOX']])

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), "toxicity_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()