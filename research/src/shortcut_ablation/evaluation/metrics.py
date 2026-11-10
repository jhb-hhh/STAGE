import numpy as np
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def macro_false_positive_rate(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fprs = []
    for cls in range(num_classes):
        true_negative = np.sum((y_true != cls) & (y_pred != cls))
        false_positive = np.sum((y_true != cls) & (y_pred == cls))
        denom = false_positive + true_negative
        if denom > 0:
            fprs.append(false_positive / denom)

    return float(np.mean(fprs)) if fprs else 0.0


def compute_metrics(y_true, y_pred, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "fpr_macro": macro_false_positive_rate(y_true, y_pred, num_classes),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
