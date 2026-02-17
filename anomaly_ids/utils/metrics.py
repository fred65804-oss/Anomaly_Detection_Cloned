import numpy as np
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    log_loss,
    confusion_matrix
)

def classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute classification metrics from probabilities
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "log_loss": log_loss(y_true, y_prob, eps=1e-15),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }
