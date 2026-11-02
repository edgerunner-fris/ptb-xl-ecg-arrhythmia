"""Multilabel metrics. Classes without both labels present get AUC=null, not 0."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, names, thr: float = 0.5) -> dict:
    per = {}
    for j, n in enumerate(names):
        yt, yp = y_true[:, j], y_prob[:, j]
        per[n] = {
            "auc": _auc(yt, yp),
            "f1": float(f1_score(yt, (yp >= thr).astype(int), zero_division=0)),
            "aucpr": float(average_precision_score(yt, yp)) if yt.sum() > 0 else float("nan"),
            "prevalence": float(yt.mean()),
            "n_pos": int(yt.sum()),
        }
    aucs = [v["auc"] for v in per.values() if not np.isnan(v["auc"])]
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_evaluable_auc": len(aucs),
        "n_classes": len(names),
        "macro_f1_at_0.5": float(f1_score(y_true, (y_prob >= thr).astype(int),
                                          average="macro", zero_division=0)),
        "per_class": per,
    }
