from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }

