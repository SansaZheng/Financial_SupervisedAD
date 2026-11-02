"""Validation-only threshold selection and held-out fraud metrics."""
import numpy as np
from sklearn.metrics import (average_precision_score, auc, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score)


def select_threshold(y, scores, objective="f1", max_fpr=0.01):
    """Search all distinct validation scores, including a predict-none candidate.

    Ties choose the higher threshold (fewer alerts). The FPR constraint is
    empirical on validation and is NOT a guarantee on future/test transactions.
    """
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Threshold selection needs two classes and finite scores")
    if objective not in {"f1", "fpr"} or not 0 <= max_fpr <= 1:
        raise ValueError("Invalid threshold objective or max_fpr")
    order = np.argsort(-scores, kind="stable")
    sy, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss)-1]
    tp = np.r_[0, np.cumsum(sy)[ends]]
    fp = np.r_[0, np.cumsum(1-sy)[ends]]
    thresholds = np.r_[np.nextafter(ss[0], np.inf), ss[ends]]
    if objective == "f1":
        utility = 2 * tp / np.maximum(tp + fp + y.sum(), 1)
    else:
        utility = np.where(fp / (len(y)-y.sum()) <= max_fpr, tp / y.sum(), -1)
    return float(thresholds[np.argmax(utility)])


def evaluate(y, scores, threshold):
    y, scores = np.asarray(y), np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, scores)
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "pr_auc_trapezoid": float(auc(recall, precision)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "fpr": float(fp / (tn + fp)), "alert_rate": float(pred.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "positive_rate": float(y.mean()), "threshold": float(threshold),
    }
