import numpy as np

from ml.features import (
    featurize,
    canonical_smiles,
    fingerprint_bits,
    descriptor_vector,
    FP_BITS,
    N_DESCRIPTORS,
    FEATURE_DIM,
)
from rdkit import Chem

from tests.conftest import ASPIRIN_SMILES, ETHANOL_SMILES


def test_featurize_shapes_and_dtype():
    combined, fp = featurize(ASPIRIN_SMILES)
    assert combined.shape == (FEATURE_DIM,)
    assert combined.dtype == np.float64
    assert fp.shape == (FP_BITS,)
    assert set(np.unique(fp)).issubset({0, 1})


def test_feature_dim_is_fingerprint_plus_descriptors():
    assert FEATURE_DIM == FP_BITS + N_DESCRIPTORS


def test_featurize_invalid_smiles_returns_none():
    assert featurize("this is not a molecule") is None
    assert featurize("Xx@@notreal") is None


def test_featurize_empty_string_is_a_degenerate_zero_vector():
    # RDKit parses "" into a *valid empty molecule* (0 atoms),
    # so featurize("") returns an all-zero vector rather than None.
    result = featurize("")
    assert result is not None
    combined, fp = result
    assert not combined.any()
    assert not fp.any()


def test_descriptor_vector_length_matches_constant():
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    assert descriptor_vector(mol).shape == (N_DESCRIPTORS,)


def test_fingerprint_bits_is_binary():
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    fp = fingerprint_bits(mol)
    assert fp.shape == (FP_BITS,)
    assert fp.dtype == np.uint8


def test_canonical_smiles_is_order_invariant():
    assert canonical_smiles("CCO") == canonical_smiles("OCC")


def test_canonical_smiles_invalid_returns_none():
    assert canonical_smiles("Xx@@notreal") is None


def test_canonical_smiles_is_idempotent():
    once = canonical_smiles(ETHANOL_SMILES)
    assert canonical_smiles(once) == once
