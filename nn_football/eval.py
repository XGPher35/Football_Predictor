import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, f1_score, mean_squared_error


def compute_class_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    # probs: (N, C), labels: (N,)
    eps = 1e-9
    ll = log_loss(labels, probs, labels=[0, 1, 2])
    # brier: multiclass brier is mean squared of one-hot vs probs
    one_hot = np.eye(probs.shape[1])[labels]
    brier = float(((one_hot - probs) ** 2).mean())
    preds = probs.argmax(axis=1)
    f1 = f1_score(labels, preds, average="macro")
    return {"log_loss": float(ll), "brier": float(brier), "f1_macro": float(f1)}


def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    # preds/targets shape: (N, k)
    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    return {"rmse": rmse}
