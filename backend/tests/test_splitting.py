import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ml.train_model import scaffold_split, tune_threshold

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccccc1",     
    "c1ccccc1C",    
    "C1CCCCC1",     
    "O=C(O)c1ccccc1",
    "CCO",
    "CCN(CC)CC",
]


def _scaffold(smi):
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=Chem.MolFromSmiles(smi), includeChirality=False
    )


def test_split_partitions_all_indices_without_overlap():
    train, test = scaffold_split(SMILES, test_size=0.3)
    assert set(train).isdisjoint(set(test))
    assert sorted(train.tolist() + test.tolist()) == list(range(len(SMILES)))


def test_no_scaffold_straddles_train_and_test():
    train, test = scaffold_split(SMILES, test_size=0.3)
    train_scaffolds = {_scaffold(SMILES[i]) for i in train}
    test_scaffolds = {_scaffold(SMILES[i]) for i in test}
    assert train_scaffolds.isdisjoint(test_scaffolds)


def test_split_is_deterministic():
    a = scaffold_split(SMILES, test_size=0.3)
    b = scaffold_split(SMILES, test_size=0.3)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_larger_scaffold_groups_go_to_train():
    train, test = scaffold_split(SMILES, test_size=0.3)
    assert len(train) >= len(test)


def test_tune_threshold_separates_clean_data():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.10, 0.20, 0.80, 0.90])
    thr = tune_threshold(y_true, y_prob)
    assert 0.20 < thr <= 0.80
    preds = (y_prob >= thr).astype(int)
    assert np.array_equal(preds, y_true)


def test_tune_threshold_returns_float():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.3, 0.6, 0.4, 0.9, 0.7, 0.2])
    assert isinstance(tune_threshold(y_true, y_prob), float)
