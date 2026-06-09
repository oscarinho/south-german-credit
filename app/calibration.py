"""Calibration and cost-sensitive threshold tuning utilities."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix


def reliability_curve(y_true, y_proba, n_bins=10):
    """Return arrays for a reliability diagram."""
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="quantile",
    )
    brier = brier_score_loss(y_true, y_proba)
    return prob_true, prob_pred, brier


def expected_cost(y_true, y_pred, cost_fn=5, cost_fp=1):
    """Total expected cost given a cost ratio on errors."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return cost_fn * fn + cost_fp * fp


def threshold_sweep(y_true, y_proba, cost_fn=5, cost_fp=1, n=101):
    """Sweep thresholds and return DataFrame with cost / recall / precision."""
    thresholds = np.linspace(0.01, 0.99, n)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        rows.append({
            "threshold": t,
            "FN": int(fn),
            "FP": int(fp),
            "cost": cost_fn * fn + cost_fp * fp,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
        })
    return pd.DataFrame(rows)


def optimal_threshold(y_true, y_proba, cost_fn=5, cost_fp=1):
    """Return the threshold minimizing expected cost."""
    sweep = threshold_sweep(y_true, y_proba, cost_fn, cost_fp)
    return sweep.loc[sweep["cost"].idxmin()]
