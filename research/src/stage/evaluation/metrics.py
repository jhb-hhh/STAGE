"""Metric utilities for STAGE evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def calculate_nids_fpr(y_true, y_pred, benign_label=0):
    """Compute the false-positive rate with benign traffic as the negative class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    benign_mask = y_true == benign_label

    if benign_mask.sum() == 0:
        return 0.0

    false_positive = np.sum(y_pred[benign_mask] != benign_label)
    return false_positive / benign_mask.sum()


def calculate_metrics_dict(y_true, y_pred, benign_label=0):
    """Compute the classification metrics used by STAGE experiments."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "FPR": calculate_nids_fpr(y_true, y_pred, benign_label=benign_label),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def format_mean_std(values, digits=4):
    """Format a sequence as mean plus standard deviation."""
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.{digits}f}±{values.std():.{digits}f}"


def summarize_fold_metrics(fold_metrics, digits=4):
    """Summarize per-fold metric dictionaries."""
    if not fold_metrics:
        return {}

    keys = fold_metrics[0].keys()
    summary = {}

    for key in keys:
        values = [metrics[key] for metrics in fold_metrics]
        summary[key] = format_mean_std(values, digits=digits)

    return summary
