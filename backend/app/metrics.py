"""Module 4 – métriques, courbes et effet du seuil de décision."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def _thin(x, y, n=200):
    if len(x) <= n:
        return list(map(float, x)), list(map(float, y))
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return [float(x[i]) for i in idx], [float(y[i]) for i in idx]


def at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float((tp + tn) / len(y)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_table(y, p, step=0.01):
    """tp/fp/fn/tn pour une grille de seuils : le curseur de l'interface en déduit tout en direct."""
    y = np.asarray(y)
    ts = np.round(np.arange(0, 1 + step / 2, step), 4)
    rows = []
    for t in ts:
        pred = p >= t
        tp = int(((pred) & (y == 1)).sum())
        fp = int(((pred) & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        tn = int(((~pred) & (y == 0)).sum())
        rows.append([float(t), tp, fp, fn, tn])
    return {"columns": ["threshold", "tp", "fp", "fn", "tn"], "rows": rows}


def evaluate(y, p, thr, with_curves=True):
    y = np.asarray(y)
    auc = float(roc_auc_score(y, p))
    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    out = {
        "auc": auc,
        "pr_auc": float(average_precision_score(y, p)),
        "gini": 2 * auc - 1,
        "ks": float(np.max(tpr - fpr)),
        **at_threshold(y, p, thr),
    }
    if with_curves:
        fx, fy = _thin(fpr, tpr)
        px, py = _thin(rc, pr)
        out["roc"] = {"fpr": fx, "tpr": fy}
        out["pr"] = {"recall": px, "precision": py}
        out["table"] = threshold_table(y, p)
    return out


def best_f1_threshold(y, p):
    ts = np.linspace(0.02, 0.98, 97)
    f = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f))])
