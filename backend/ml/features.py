import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors

# Morgan fingerprint settings.
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
    """Morgan fingerprint as a (FP_BITS,) uint8 array of 0/1."""
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)


def descriptor_vector(mol):
    """Physicochemical descriptors as a (N_DESCRIPTORS,) float array."""
    return np.array([fn(mol) for _, fn in DESCRIPTORS], dtype=np.float64)


def featurize(smiles):
    """Return (combined_features, fingerprint_bits) for a SMILES, or None.

    combined_features : (FEATURE_DIM,) float, model input.
    fingerprint_bits  : (FP_BITS,)   uint8, used for applicability-domain.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = fingerprint_bits(mol)
    desc = descriptor_vector(mol)
    combined = np.concatenate([fp.astype(np.float64), desc])
    return combined, fp


def canonical_smiles(smiles):
    """Canonical SMILES for de-duplication, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)
