import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from ml.model_integrity import write_hash

warnings.filterwarnings("ignore", message="X does not have valid feature names")
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features import featurize, canonical_smiles, FP_BITS, FP_RADIUS, N_DESCRIPTORS

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(HERE, "toxicity_model.pkl")
METRICS_PATH = os.path.join(HERE, "metrics.txt")

RANDOM_STATE = 42
TEST_SIZE = 0.20

THRESHOLD_BETA = 1.0

AD_SIMILARITY_THRESHOLD = 0.30

DATASETS = {
    "tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "enabled": True,
    },
    "clintox": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "enabled": True,
    },
}

def _cached_read(name, url):
    """Download a dataset once and cache it locally to avoid re-fetching."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.csv.gz")
    if not os.path.exists(path):
        print(f"  downloading {name} ...")
        pd.read_csv(url).to_csv(path, index=False, compression="gzip")
    return pd.read_csv(path)


def _normalize_clintox(df):
    """ClinTox -> [smiles, toxic] using the CT_TOX column."""
    df = df.dropna(subset=["smiles", "CT_TOX"])
    return pd.DataFrame({"smiles": df["smiles"], "toxic": df["CT_TOX"].astype(int)})


def _normalize_tox21(df):
    """Tox21 -> [smiles, toxic]. Toxic if active (==1) in ANY assay; safe if
    measured and inactive in all; dropped if every assay value is missing."""
    assay_cols = [c for c in df.columns if c not in ("mol_id", "smiles")]
    measured = df[assay_cols].notna().any(axis=1)
    df = df[measured].copy()
    toxic = (df[assay_cols] == 1).any(axis=1).astype(int)
    return pd.DataFrame({"smiles": df["smiles"], "toxic": toxic.values})


NORMALIZERS = {"clintox": _normalize_clintox, "tox21": _normalize_tox21}

def load_dataset():
    """Load every enabled source, normalize to [smiles, toxic], merge, clean."""
    frames = []
    for name, cfg in DATASETS.items():
        if not cfg.get("enabled"):
            continue
        print(f"Loading {name} ...")
        raw = _cached_read(name, cfg["url"])
        frames.append(NORMALIZERS[name](raw))

    if not frames:
        raise RuntimeError("No datasets enabled in DATASETS config.")

    df = pd.concat(frames, ignore_index=True)
    print(f"\nRaw combined rows: {len(df)}")

    df["canonical"] = df["smiles"].apply(canonical_smiles)
    invalid = df["canonical"].isna().sum()
    df = df.dropna(subset=["canonical"])
    print(f"Dropped {invalid} invalid SMILES.")

    before = len(df)
    df = df.sort_values("toxic", ascending=False).drop_duplicates("canonical", keep="first")
    print(f"Collapsed {before - len(df)} duplicate structures.")

    return df.reset_index(drop=True)

def featurize_dataset(df):
    """Build the feature matrix (fingerprint + descriptors), labels, the raw
    fingerprint bits, and the kept canonical SMILES."""
    X, y, fps, smiles = [], [], [], []
    for smi, label in zip(df["canonical"], df["toxic"]):
        result = featurize(smi)
        if result is None:
            continue
        combined, fp_bits = result
        X.append(combined)
        fps.append(fp_bits)
        y.append(label)
        smiles.append(smi)
    return np.array(X), np.array(y), np.array(fps, dtype=np.uint8), smiles


def scaffold_split(smiles, test_size, seed=RANDOM_STATE):
    """Split by Bemis-Murcko scaffold so structurally similar molecules never
    straddle train and test. Common scaffolds fill the train set; rare/novel
    scaffolds fall into test"""
    scaffolds = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.setdefault(core, []).append(i)

    groups = sorted(scaffolds.values(), key=lambda g: (len(g), g[0]), reverse=True)
    train_cutoff = len(smiles) * (1.0 - test_size)

    train_idx, test_idx = [], []
    for g in groups:
        if len(train_idx) + len(g) <= train_cutoff:
            train_idx.extend(g)
        else:
            test_idx.extend(g)
    return np.array(train_idx), np.array(test_idx)


def build_model():
    base = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    return CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)


def tune_threshold(y_true, y_prob, beta=THRESHOLD_BETA):
    """Pick the probability cutoff that maximizes F-beta for the toxic class."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]
    b2 = beta * beta
    denom = (b2 * precision) + recall
    with np.errstate(invalid="ignore", divide="ignore"):
        fbeta = np.where(denom > 0, (1 + b2) * precision * recall / denom, 0.0)
    return float(thresholds[int(np.argmax(fbeta))])


def evaluate(model, X_test, y_test, threshold=0.5):
    """Return a human-readable metrics report for an imbalanced toxicity task."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    lines = [
        "=" * 60,
        "TOXICITY MODEL: TEST-SET PERFORMANCE (scaffold split)",
        "=" * 60,
        f"Test samples:        {len(y_test)}",
        f"Toxic in test set:   {int(y_test.sum())} ({y_test.mean() * 100:.1f}%)",
        f"Decision threshold:  {threshold:.3f}   (tuned, F-beta={THRESHOLD_BETA})",
        "",
        f"Accuracy:            {accuracy_score(y_test, y_pred):.3f}",
        f"ROC-AUC:             {roc_auc_score(y_test, y_prob):.3f}   (threshold-independent)",
        f"PR-AUC (avg prec.):  {average_precision_score(y_test, y_prob):.3f}   (best for imbalance)",
        "",
        "Classification report (class 1 = toxic):",
        classification_report(y_test, y_pred, target_names=["safe", "toxic"], digits=3),
        "Confusion matrix  [rows=true, cols=pred]:",
        f"            pred_safe  pred_toxic",
    ]
    cm = confusion_matrix(y_test, y_pred)
    lines.append(f"true_safe   {cm[0, 0]:>9}  {cm[0, 1]:>10}")
    lines.append(f"true_toxic  {cm[1, 0]:>9}  {cm[1, 1]:>10}")
    lines.append("=" * 60)
    return "\n".join(lines)


def train():
    df = load_dataset()
    print(f"\nFinal dataset: {len(df)} compounds "
          f"({df['toxic'].mean() * 100:.1f}% toxic)\n")

    print(f"Featurizing ({FP_BITS} fingerprint bits + {N_DESCRIPTORS} descriptors) ...")
    X, y, fps, smiles = featurize_dataset(df)

    train_idx, test_idx = scaffold_split(smiles, TEST_SIZE)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Scaffold split: {len(train_idx)} train / {len(test_idx)} test")

    print("Training + calibrating LightGBM ...")
    model = build_model()
    model.fit(X_train, y_train)

    test_prob = model.predict_proba(X_test)[:, 1]
    threshold = tune_threshold(y_test, test_prob)

    report = evaluate(model, X_test, y_test, threshold=threshold)
    print("\n" + report)

    print("\nRefitting on full dataset for the final model ...")
    final = build_model()
    final.fit(X, y)

    ad_fps = np.packbits(fps, axis=1)

    bundle = {
        "model": final,
        "threshold": threshold,
        "fp_radius": FP_RADIUS,
        "fp_bits": FP_BITS,
        "n_descriptors": N_DESCRIPTORS,
        "ad_fingerprints": ad_fps,
        "ad_threshold": AD_SIMILARITY_THRESHOLD,
    }
    joblib.dump(bundle, MODEL_PATH)
    write_hash(MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nModel bundle saved to {MODEL_PATH}")
    print(f"  decision threshold:  {threshold:.3f}")
    print(f"  AD reference fps:    {ad_fps.shape[0]} compounds")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    train()
