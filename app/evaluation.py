"""Evaluation metrics and plots."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve,
)


def evaluate_model(pipeline, X, y):
    y_pred = pipeline.predict(X)
    try:
        y_proba = pipeline.predict_proba(X)[:, 1]
    except Exception:
        y_proba = None
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba) if y_proba is not None else np.nan,
    }


def print_evaluation_report(pipeline, X, y, title="Evaluation"):
    m = evaluate_model(pipeline, X, y)
    print("=" * 50)
    print(title)
    print("=" * 50)
    print(f"Accuracy:  {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall:    {m['recall']:.4f}")
    print(f"F1 Score:  {m['f1']:.4f}")
    print(f"ROC AUC:   {m['roc_auc']:.4f}")
    print("=" * 50)
    y_pred = pipeline.predict(X)
    print("\nClassification Report:")
    print(classification_report(
        y, y_pred, target_names=["Good Credit", "Bad Credit"], zero_division=0,
    ))
    return m


def plot_confusion_matrix(pipeline, X, y, ax=None):
    y_pred = pipeline.predict(X)
    cm = confusion_matrix(y, y_pred)
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0],
                xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
    ax[0].set_title("Confusion Matrix")
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax[1],
                xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
    ax[1].set_title("Normalized")
    ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual")
    plt.tight_layout()
    plt.show()
    return cm


def plot_roc_curve(pipeline, X, y):
    y_proba = pipeline.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Set")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    return auc


def plot_model_comparison_boxplot(results, metrics=("accuracy", "recall", "f1")):
    """Boxplot of CV test scores across models, one subplot per metric."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    names = list(results.keys())
    for ax, metric in zip(axes, metrics):
        data = [results[name][f"test_{metric}"] for name in names]
        ax.boxplot(data, labels=names, showmeans=True)
        ax.set_title(metric.capitalize())
        ax.set_ylabel(metric.capitalize())
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def summarize_cv_results(results):
    """Return a tidy DataFrame summarizing CV results."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{r['test_accuracy'].mean():.4f} ± {r['test_accuracy'].std():.4f}",
            "Precision": f"{r['test_precision'].mean():.4f} ± {r['test_precision'].std():.4f}",
            "Recall": f"{r['test_recall'].mean():.4f} ± {r['test_recall'].std():.4f}",
            "F1": f"{r['test_f1'].mean():.4f} ± {r['test_f1'].std():.4f}",
            "ROC AUC": f"{r['test_roc_auc'].mean():.4f} ± {r['test_roc_auc'].std():.4f}",
        })
    return pd.DataFrame(rows)
