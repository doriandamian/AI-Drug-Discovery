import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors

FP_RADIUS = 2
FP_BITS = 2048

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("RingCount", Descriptors.RingCount),
]
N_DESCRIPTORS = len(DESCRIPTORS)
FEATURE_DIM = FP_BITS + N_DESCRIPTORS


def fingerprint_bits(mol):
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)


def descriptor_vector(mol):
    return np.array([fn(mol) for _, fn in DESCRIPTORS], dtype=np.float64)


def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = fingerprint_bits(mol)
    desc = descriptor_vector(mol)
    combined = np.concatenate([fp.astype(np.float64), desc])
    return combined, fp


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)
