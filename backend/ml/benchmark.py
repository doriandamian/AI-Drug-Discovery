import os
import sys
import warnings
import numpy as np
import joblib
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features import featurize, FP_BITS, N_DESCRIPTORS
from ml.train_model import scaffold_split, TEST_SIZE
from ml.train_multitask import load_multitask, MODEL_PATH


def held_out_evaluation():
    bundle = joblib.load(MODEL_PATH)
    tasks = bundle["tasks"]

    master, _ = load_multitask()
    smiles = master["canonical"].tolist()

    train_idx, test_idx = scaffold_split(smiles, TEST_SIZE)
    test_set = set(test_idx.tolist())

    X = np.zeros((len(smiles), FP_BITS + N_DESCRIPTORS), dtype=np.float64)
    for i, smi in enumerate(smiles):
        X[i] = featurize(smi)[0]

    is_test = np.array([i in test_set for i in range(len(smiles))])

    out = []
    out.append("=" * 78)
    out.append("PART 1: HELD-OUT EVALUATION (scaffold split, models never saw these)")
    out.append("=" * 78)
    out.append(f"{'Endpoint':<15}{'N_test':>7}{'ROC-AUC':>9}{'PR-AUC':>8}"
               f"{'Prec':>7}{'Recall':>8}{'F1':>7}")
    out.append("-" * 78)

    pooled_true, pooled_prob, pooled_pred = [], [], []
    roc_list, pr_list, f1_list = [], [], []

    for name, t in tasks.items():
        labels = master[name].values.astype(float)
        mask = is_test & ~np.isnan(labels)
        y_true = labels[mask].astype(int)
        if y_true.sum() == 0 or (y_true == 0).sum() == 0:
            continue

        prob = t["model"].predict_proba(X[mask])[:, 1]
        pred = (prob >= t["threshold"]).astype(int)

        roc = roc_auc_score(y_true, prob)
        pr = average_precision_score(y_true, prob)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        roc_list.append(roc); pr_list.append(pr); f1_list.append(f1)
        pooled_true.extend(y_true); pooled_prob.extend(prob); pooled_pred.extend(pred)

        out.append(f"{name:<15}{len(y_true):>7}{roc:>9.3f}{pr:>8.3f}"
                   f"{prec:>7.3f}{rec:>8.3f}{f1:>7.3f}")

    out.append("-" * 78)
    pooled_true = np.array(pooled_true)
    pooled_prob = np.array(pooled_prob)
    pooled_pred = np.array(pooled_pred)

    out.append(f"{'MACRO AVG':<15}{'':>7}{np.mean(roc_list):>9.3f}{np.mean(pr_list):>8.3f}"
               f"{'':>7}{'':>8}{np.mean(f1_list):>7.3f}")
    out.append(f"{'MICRO/POOLED':<15}{len(pooled_true):>7}"
               f"{roc_auc_score(pooled_true, pooled_prob):>9.3f}"
               f"{average_precision_score(pooled_true, pooled_prob):>8.3f}"
               f"{precision_score(pooled_true, pooled_pred, zero_division=0):>7.3f}"
               f"{recall_score(pooled_true, pooled_pred, zero_division=0):>8.3f}"
               f"{f1_score(pooled_true, pooled_pred, zero_division=0):>7.3f}")
    out.append("=" * 78)
    report = "\n".join(out)
    print(report)
    return report, {
        "macro_roc": float(np.mean(roc_list)),
        "macro_pr": float(np.mean(pr_list)),
        "pooled_roc": float(roc_auc_score(pooled_true, pooled_prob)),
        "pooled_acc": float(accuracy_score(pooled_true, pooled_pred)),
        "pooled_prec": float(precision_score(pooled_true, pooled_pred, zero_division=0)),
        "pooled_rec": float(recall_score(pooled_true, pooled_pred, zero_division=0)),
        "n_endpoints": len(roc_list),
    }


PANEL = [
    ("Bisphenol A",       "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",          "NR-ER"),
    ("Diethylstilbestrol","CC/C(=C(\\CC)/c1ccc(O)cc1)/c1ccc(O)cc1",  "NR-ER"),
    ("17b-Estradiol",     "CC12CCC3C(CCc4cc(O)ccc34)C1CCC2O",        "NR-ER"),
    ("Genistein",         "O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12",    "NR-ER"),
    ("Rotenone",          "CC(=C)C1Cc2c(ccc3c2OC2COc4cc(OC)c(OC)cc4C2C3=O)O1", "SR-MMP"),
    ("Thalidomide",       "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",     "ClinTox"),
    ("Caffeine",          "Cn1cnc2c1c(=O)n(C)c(=O)n2C",              None),
    ("Glucose",           "OCC1OC(O)C(O)C(O)C1O",                    None),
    ("Glycine",           "NCC(=O)O",                                None),
    ("Ethanol",           "CCO",                                     None),
    ("Ascorbic acid",     "OCC(O)C1OC(=O)C(O)=C1O",                  None),
    ("Aspirin",           "CC(=O)Oc1ccccc1C(=O)O",                   None),
]

STRONG = 0.50


def named_demonstration():
    from tools.toxicity_predictor import _BUNDLE
    tasks = _BUNDLE["tasks"]

    print("\n" + "=" * 78)
    print("PART 2: NAMED-COMPOUND DEMONSTRATION (illustrative; some are in training)")
    print("=" * 78)

    hits, total_expected = 0, 0
    benign_clean, benign_total = 0, 0

    for name, smi, expected in PANEL:
        feat = featurize(smi)
        if feat is None:
            print(f"{name:<20} INVALID SMILES")
            continue
        x = feat[0].reshape(1, -1)
        probs = {tn: float(t["model"].predict_proba(x)[0][1]) for tn, t in tasks.items()}
        top_name = max(probs, key=probs.get)
        top_p = probs[top_name]

        if expected is not None:
            total_expected += 1
            ep = probs.get(expected, float("nan"))
            ok = ep >= STRONG
            hits += int(ok)
            mark = "✓" if ok else "✗"
            print(f"{name:<20} expect {expected:<10} -> {ep*100:>3.0f}%  {mark}   "
                  f"(top: {top_name} {top_p*100:.0f}%)")
        else:
            benign_total += 1
            strong_flags = [tn for tn, p in probs.items() if p >= STRONG]
            clean = len(strong_flags) == 0
            benign_clean += int(clean)
            mark = "✓ clear" if clean else f"✗ flags: {', '.join(strong_flags)}"
            print(f"{name:<20} expect clear      -> top {top_name} {top_p*100:.0f}%   {mark}")

    print("-" * 78)
    print(f"Known-mechanism recall (≥{int(STRONG*100)}% on expected endpoint): "
          f"{hits}/{total_expected}")
    print(f"Benign compounds with no confident flag:                 "
          f"{benign_clean}/{benign_total}")
    print("=" * 78)
    return hits, total_expected, benign_clean, benign_total


REPORT_PATH = os.path.join(HERE if (HERE := os.path.dirname(__file__)) else ".", "benchmark_report.txt")

if __name__ == "__main__":
    table, metrics = held_out_evaluation()
    hits, n_exp, clean, n_ben = named_demonstration()

    verdict = "\n".join([
        "#" * 78,
        "FINAL VERDICT",
        "#" * 78,
        f"Held-out endpoints evaluated:  {metrics['n_endpoints']}",
        f"Macro ROC-AUC:                 {metrics['macro_roc']:.3f}",
        f"Pooled ROC-AUC:                {metrics['pooled_roc']:.3f}",
        f"Pooled precision / recall:     {metrics['pooled_prec']:.3f} / {metrics['pooled_rec']:.3f}",
        f"Sanity panel, mechanism hits: {hits}/{n_exp}",
        f"Sanity panel, benign clean:   {clean}/{n_ben}",
        "#" * 78,
    ])
    print("\n" + verdict)

    with open(REPORT_PATH, "w") as f:
        f.write(table + "\n\n" + verdict + "\n")
    print(f"\nSaved benchmark report to {REPORT_PATH}")
