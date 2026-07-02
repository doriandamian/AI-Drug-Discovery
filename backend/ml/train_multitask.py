"""Multi-task toxicity training.

Instead of collapsing every assay into one binary "toxic" label, this trains a
separate calibrated classifier for EACH toxicity endpoint (the 12 Tox21 assays
plus ClinTox clinical-trial toxicity). The sparse label matrix is handled
naturally: each task trains only on the molecules that have a label for it.

A single GLOBAL scaffold split assigns molecules to train/test once, so no
scaffold leaks across tasks and the per-task scores stay comparable.

Output: tools/toxicity_predictor.py consumes the saved bundle and returns a
per-endpoint toxicity profile rather than a single yes/no.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from ml.model_integrity import write_hash
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features import featurize, canonical_smiles, FP_BITS, FP_RADIUS, N_DESCRIPTORS
from ml.train_model import (
    _cached_read,
    scaffold_split,
    tune_threshold,
    DATASETS,
    RANDOM_STATE,
    TEST_SIZE,
    AD_SIMILARITY_THRESHOLD,
)

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "toxicity_model.pkl")
METRICS_PATH = os.path.join(HERE, "metrics.txt")

MIN_PER_CLASS = 40

TASK_INFO = {
    "NR-AR": "Androgen receptor (endocrine disruption)",
    "NR-AR-LBD": "Androgen receptor, ligand-binding domain",
    "NR-AhR": "Aryl hydrocarbon receptor",
    "NR-Aromatase": "Aromatase enzyme inhibition",
    "NR-ER": "Estrogen receptor (endocrine disruption)",
    "NR-ER-LBD": "Estrogen receptor, ligand-binding domain",
    "NR-PPAR-gamma": "PPAR-gamma nuclear receptor",
    "SR-ARE": "Oxidative stress (antioxidant response element)",
    "SR-ATAD5": "Genotoxicity / DNA damage (ATAD5)",
    "SR-HSE": "Heat-shock stress response",
    "SR-MMP": "Mitochondrial toxicity (membrane potential)",
    "SR-p53": "DNA-damage response (p53)",
    "ClinTox": "Clinical-trial toxicity failure",
}


def load_multitask():
    """Build a master table: one row per unique molecule, one column per task,
    values in {0, 1, NaN}. NaN means 'not measured for this endpoint'."""
    tox = _cached_read("tox21", DATASETS["tox21"]["url"])
    assay_cols = [c for c in tox.columns if c not in ("mol_id", "smiles")]
    tox["canonical"] = tox["smiles"].apply(canonical_smiles)
    tox = tox.dropna(subset=["canonical"])
    tox = tox.groupby("canonical")[assay_cols].max().reset_index()

    clin = _cached_read("clintox", DATASETS["clintox"]["url"])
    clin["canonical"] = clin["smiles"].apply(canonical_smiles)
    clin = clin.dropna(subset=["canonical", "CT_TOX"])
    clin = (
        clin.groupby("canonical")["CT_TOX"].max().reset_index()
        .rename(columns={"CT_TOX": "ClinTox"})
    )

    master = pd.merge(tox, clin, on="canonical", how="outer")
    tasks = assay_cols + ["ClinTox"]
    return master, tasks


def build_task_model():
    base = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=48,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    return CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)


def train():
    master, tasks = load_multitask()
    print(f"Loaded {len(master)} unique molecules across {len(tasks)} endpoints.\n")

    print(f"Featurizing ({FP_BITS} fingerprint bits + {N_DESCRIPTORS} descriptors) ...")
    smiles = master["canonical"].tolist()
    X = np.zeros((len(smiles), FP_BITS + N_DESCRIPTORS), dtype=np.float64)
    fps = np.zeros((len(smiles), FP_BITS), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        combined, fp_bits = featurize(smi)
        X[i] = combined
        fps[i] = fp_bits

    train_idx, test_idx = scaffold_split(smiles, TEST_SIZE)
    train_mask = np.zeros(len(smiles), dtype=bool)
    train_mask[train_idx] = True
    test_mask = np.zeros(len(smiles), dtype=bool)
    test_mask[test_idx] = True
    print(f"Global scaffold split: {len(train_idx)} train / {len(test_idx)} test\n")

    trained = {}
    report_rows = []
    for task in tasks:
        labels = master[task].values.astype(float)
        labeled = ~np.isnan(labels)

        tr = train_mask & labeled
        te = test_mask & labeled
        y_tr = labels[tr].astype(int)

        pos, neg = int(y_tr.sum()), int((y_tr == 0).sum())
        if pos < MIN_PER_CLASS or neg < MIN_PER_CLASS:
            print(f"  skip {task}: too few labels (pos={pos}, neg={neg})")
            continue

        model = build_task_model()
        model.fit(X[tr], y_tr)

        y_te = labels[te].astype(int)
        if y_te.sum() == 0 or (y_te == 0).sum() == 0:
            threshold, roc, prauc, rec, n_te = 0.5, float("nan"), float("nan"), float("nan"), int(te.sum())
        else:
            prob = model.predict_proba(X[te])[:, 1]
            threshold = tune_threshold(y_te, prob)
            roc = roc_auc_score(y_te, prob)
            prauc = average_precision_score(y_te, prob)
            rec = recall_score(y_te, (prob >= threshold).astype(int))
            n_te = len(y_te)

        trained[task] = {
            "model": model,
            "threshold": threshold,
            "n_train": int(tr.sum()),
            "n_test": n_te,
            "roc_auc": roc,
            "pr_auc": prauc,
            "recall": rec,
        }
        report_rows.append((task, pos + neg, roc, prauc, rec, threshold))
        print(f"  trained {task:<14} train={tr.sum():>5}  ROC-AUC={roc:.3f}  PR-AUC={prauc:.3f}")

    ad_fps = np.packbits(fps[train_idx], axis=1)

    bundle = {
        "tasks": trained,
        "task_info": TASK_INFO,
        "fp_radius": FP_RADIUS,
        "fp_bits": FP_BITS,
        "n_descriptors": N_DESCRIPTORS,
        "ad_fingerprints": ad_fps,
        "ad_threshold": AD_SIMILARITY_THRESHOLD,
    }
    joblib.dump(bundle, MODEL_PATH)
    write_hash(MODEL_PATH)

    report = _format_report(report_rows)
    print("\n" + report)
    with open(METRICS_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nMulti-task bundle saved to {MODEL_PATH}")
    print(f"  endpoints trained:  {len(trained)}")
    print(f"  AD reference fps:   {ad_fps.shape[0]} compounds")


def _format_report(rows):
    macro_roc = np.nanmean([r[2] for r in rows]) if rows else float("nan")
    macro_pr = np.nanmean([r[3] for r in rows]) if rows else float("nan")
    lines = [
        "=" * 72,
        "MULTI-TASK TOXICITY MODEL: PER-ENDPOINT PERFORMANCE (scaffold split)",
        "=" * 72,
        f"{'Endpoint':<16}{'N':>7}{'ROC-AUC':>10}{'PR-AUC':>9}{'Recall':>9}{'Cutoff':>9}",
        "-" * 72,
    ]
    for task, n, roc, pr, rec, thr in rows:
        lines.append(f"{task:<16}{n:>7}{roc:>10.3f}{pr:>9.3f}{rec:>9.3f}{thr:>9.3f}")
    lines.append("-" * 72)
    lines.append(f"{'MACRO AVG':<16}{'':>7}{macro_roc:>10.3f}{macro_pr:>9.3f}")
    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":
    train()
