"""Fairness audit utilities for binary classification.

Computes per-subgroup metrics so we can spot disparate impact across
sensitive attributes like personal_status_sex and foreign_worker.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def per_group_metrics(y_true, y_pred, sensitive_values):
    """Return per-group selection rate, TPR, FPR, precision, accuracy.

    Parameters
    ----------
    y_true, y_pred : 1d array-like of {0, 1}
    sensitive_values : 1d array-like with the subgroup label per sample.

    Returns
    -------
    DataFrame indexed by subgroup with: n, positives, selection_rate,
    TPR (recall), FPR, precision, accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    s = np.asarray(sensitive_values)
    rows = []
    for g in pd.unique(s):
        mask = s == g
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        n = len(yt)
        rows.append({
            "group": g,
            "n": n,
            "positives": int(yt.sum()),
            "selection_rate": (yp == 1).mean(),
            "TPR": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
            "accuracy": (yt == yp).mean(),
        })
    return pd.DataFrame(rows).set_index("group")


def disparate_impact(metrics_df, reference_group=None):
    """Disparate impact = selection_rate(group) / selection_rate(reference).

    By default the reference is the group with the highest selection rate.
    Values < 0.80 are commonly considered evidence of adverse impact.
    """
    sr = metrics_df["selection_rate"]
    if reference_group is None:
        reference_group = sr.idxmax()
    ratio = sr / sr.loc[reference_group]
    return ratio.rename(f"disparate_impact_vs_{reference_group}")


def equalized_odds_gap(metrics_df):
    """Max - min across groups for TPR and FPR. Smaller is fairer."""
    return {
        "TPR_gap": metrics_df["TPR"].max() - metrics_df["TPR"].min(),
        "FPR_gap": metrics_df["FPR"].max() - metrics_df["FPR"].min(),
    }
