"""Precompute all heavy artifacts for the Streamlit dashboard.

Runs the full CRISP-ML analysis once and caches results to ``app/artifacts/``:
- ``results.json``  — scalar metrics, tables, arrays
- ``fig_*.png``     — figures rendered headless (matplotlib Agg)

The Streamlit app then loads these instantly instead of recomputing on every
visit. Re-run with::

    python -m app.precompute
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from app.config import RANDOM_STATE
from app.data_loader import (
    load_and_prepare_data,
    split_features_target,
    create_data_splits,
    get_train_val_combined,
)
from app.preprocessing import get_feature_names
from app.models import get_best_model
from app.training import (
    compare_models,
    compare_resamplers,
    create_training_pipeline,
    cross_validate_model,
    train_best_model,
)
from app.evaluation import evaluate_model
from app.fairness import per_group_metrics, disparate_impact, equalized_odds_gap
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")

ART = Path(__file__).resolve().parent / "artifacts"
ART.mkdir(exist_ok=True)

# Shared palette (matches the app theme)
C = {
    "good": "#43936C", "bad": "#D96B5F", "gold": "#C9A86A",
    "blue": "#4A67B0", "graphite": "#2A3038", "mist": "#B0B4B8",
}


def _mean_std(cv, metric):
    s = cv[f"test_{metric}"]
    return float(s.mean()), float(s.std())


def _cv_table(results):
    """{name: cv_result} -> list of dict rows with mean/std per metric."""
    rows = []
    for name, cv in results.items():
        row = {"model": name}
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mean, std = _mean_std(cv, m)
            row[f"{m}_mean"] = round(mean, 3)
            row[f"{m}_std"] = round(std, 3)
        rows.append(row)
    return rows


def _box(results, metric, title, path, color):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    data = [results[n][f"test_{metric}"] for n in results]
    ax.boxplot(data, labels=list(results.keys()), showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(metric.capitalize())
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    out = {}

    # ----- Data -----
    df = load_and_prepare_data()
    X, y = split_features_target(df)
    counts = df["credit_risk"].value_counts().sort_index()
    out["target"] = {"good": int(counts[0]), "bad": int(counts[1])}
    out["imbalance_ratio"] = round(counts[0] / counts[1], 2)

    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
    X_trainval, y_trainval = get_train_val_combined(X_train, X_val, y_train, y_val)
    out["splits"] = {
        "train": int(len(y_train)), "val": int(len(y_val)),
        "test": int(len(y_test)), "trainval": int(len(y_trainval)),
    }

    # ----- EDA figures -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    counts.plot(kind="bar", ax=axes[0], color=[C["good"], C["bad"]])
    axes[0].set_title("Credit Risk Distribution")
    axes[0].set_xticklabels(["Good (0)", "Bad (1)"], rotation=0)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 8, str(v), ha="center")
    for ax, col in zip(axes[1:], ["duration", "amount"]):
        ax.hist(df[col], bins=30, color=C["gold"], edgecolor="white")
        ax.set_title(f"Distribution of {col}")
    fig.tight_layout(); fig.savefig(ART / "fig_eda.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, col in zip(axes.flat, ["status", "credit_history", "savings", "purpose"]):
        rate = df.groupby(col)["credit_risk"].mean().sort_values()
        rate.plot(kind="bar", ax=ax, color=C["gold"])
        ax.axhline(df["credit_risk"].mean(), color="black", lw=1, ls="--")
        ax.set_title(f"Bad-credit rate by {col}")
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(ART / "fig_badrate.png", dpi=110); plt.close(fig)

    # ----- Modeling: baseline (no resampling) -----
    baseline = compare_models(X_trainval, y_trainval, include_resampler=False)
    out["baseline_table"] = _cv_table(baseline)
    _box(baseline, "recall", "Recall by model — no resampling",
         ART / "fig_baseline_box.png", C["bad"])

    # ----- Resampler comparison (tuned XGB) -----
    resamplers = compare_resamplers(X_trainval, y_trainval)
    out["resampler_table"] = _cv_table(resamplers)
    _box(resamplers, "recall", "Recall by resampler — tuned XGBoost",
         ART / "fig_resampler_box.png", C["blue"])

    # ----- All models x SMOTETomek -----
    smt_models = compare_models(X_trainval, y_trainval,
                                include_resampler=True, resampler_name="SMOTETomek")
    out["smotetomek_table"] = _cv_table(smt_models)
    _box(smt_models, "f1", "F1 by model — SMOTETomek",
         ART / "fig_smotetomek_box.png", C["good"])

    # ----- Final model: stable CV -----
    final_cv_pipe = create_training_pipeline(
        get_best_model(), resampler=SMOTETomek(random_state=RANDOM_STATE))
    cv_final = cross_validate_model(final_cv_pipe, X_trainval, y_trainval)
    out["cv_final"] = {
        m: {"mean": round(_mean_std(cv_final, m)[0], 3),
            "std": round(_mean_std(cv_final, m)[1], 3)}
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    # ----- Final model: fit + test evaluation -----
    best = train_best_model(X_trainval, y_trainval, apply_resampling=True)
    test_metrics = evaluate_model(best, X_test, y_test)
    out["test_metrics"] = {k: round(float(v), 3) for k, v in test_metrics.items()}
    y_pred = best.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    out["confusion_matrix"] = cm.tolist()

    # confusion matrix figure
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Good", "Bad"]); ax.set_yticklabels(["Good", "Bad"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(ART / "fig_confusion.png", dpi=110); plt.close(fig)

    # ROC figure
    y_proba = best.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color=C["blue"], lw=2,
            label=f"ROC (AUC={test_metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ART / "fig_roc.png", dpi=110); plt.close(fig)

    # ----- Split variance (100 re-draws of val/test) -----
    X_tr_fix, X_tv_fix, y_tr_fix, y_tv_fix = train_test_split(
        X, y, train_size=0.70, shuffle=True,
        random_state=RANDOM_STATE, stratify=y)
    recalls = []
    for seed in range(100):
        Xv, Xte, yv, yte = train_test_split(
            X_tv_fix, y_tv_fix, train_size=0.5, shuffle=True,
            random_state=seed, stratify=y_tv_fix)
        Xt = pd.concat([X_tr_fix, Xv], ignore_index=True)
        yt = pd.concat([y_tr_fix, yv], ignore_index=True)
        p = train_best_model(Xt, yt, apply_resampling=True)
        recalls.append(evaluate_model(p, Xte, yte)["recall"])
    recalls = np.array(recalls)
    master_recall, seeded_recall = 0.556, float(test_metrics["recall"])
    pct = float((recalls < master_recall).mean() * 100)
    out["split_variance"] = {
        "mean": round(float(recalls.mean()), 3),
        "std": round(float(recalls.std()), 3),
        "p5": round(float(np.percentile(recalls, 5)), 3),
        "p95": round(float(np.percentile(recalls, 95)), 3),
        "master_recall": master_recall,
        "seeded_recall": round(seeded_recall, 3),
        "master_percentile": round(pct, 0),
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(recalls, bins=15, color="#6FA8DC", edgecolor="white")
    ax.axvline(recalls.mean(), color="black", ls="--", lw=1.5,
               label=f"mean ({recalls.mean():.3f})")
    ax.axvline(master_recall, color=C["bad"], lw=2,
               label=f"master's reported ({master_recall:.3f})")
    ax.axvline(seeded_recall, color=C["good"], lw=2,
               label=f"this app, seed=42 ({seeded_recall:.3f})")
    ax.set_xlabel("Test-set recall"); ax.set_ylabel("Number of splits")
    ax.set_title("Recall is split-dependent (150-row test = high variance)")
    ax.legend(); fig.tight_layout()
    fig.savefig(ART / "fig_split_variance.png", dpi=110); plt.close(fig)

    # ----- Permutation importance -----
    pre = best.named_steps["ct"]; clf = best.named_steps["clf"]
    X_test_t = pre.transform(X_test)
    feat = get_feature_names(pre) or [f"f{i}" for i in range(X_test_t.shape[1])]
    perm = permutation_importance(clf, X_test_t, y_test, scoring="f1",
                                  n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1)
    perm_df = (pd.DataFrame({"feature": feat, "importance": perm.importances_mean})
               .sort_values("importance", ascending=False).head(15))
    out["permutation_top"] = perm_df.to_dict("records")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(perm_df["feature"][::-1], perm_df["importance"][::-1], color=C["gold"])
    ax.set_title("Permutation importance (test, F1 drop)")
    fig.tight_layout(); fig.savefig(ART / "fig_perm.png", dpi=110); plt.close(fig)

    # ----- SHAP (native TreeSHAP via pred_contribs) -----
    booster = clf.get_booster()
    dmat = xgb.DMatrix(X_test_t, feature_names=list(feat))
    contribs = booster.predict(dmat, pred_contribs=True)
    shap_values = contribs[:, :-1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df = (pd.DataFrame({"feature": feat, "mean_abs_shap": mean_abs})
               .sort_values("mean_abs_shap", ascending=False).head(15))
    out["shap_top"] = shap_df.round(4).to_dict("records")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(shap_df["feature"][::-1], shap_df["mean_abs_shap"][::-1], color=C["blue"])
    ax.set_title("SHAP feature importance (mean |SHAP|)")
    fig.tight_layout(); fig.savefig(ART / "fig_shap_bar.png", dpi=110); plt.close(fig)

    # ----- Fairness -----
    out["fairness"] = {}
    for attr in ["personal_status_sex", "foreign_worker"]:
        m = per_group_metrics(y_test, y_pred, X_test[attr])
        di = disparate_impact(m)
        eo = equalized_odds_gap(m)
        tbl = m.round(3).reset_index().to_dict("records")
        out["fairness"][attr] = {
            "table": tbl,
            "disparate_impact": {str(k): round(float(v), 3) for k, v in di.items()},
            "min_disparate_impact": round(float(di.min()), 3),
            "tpr_gap": round(float(eo["TPR_gap"]), 3),
            "fpr_gap": round(float(eo["FPR_gap"]), 3),
        }

    # ----- Cost-optimal threshold (FN:FP = 5:1) on test -----
    COST_FN, COST_FP = 5, 1
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = []
    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, yp, labels=[0, 1]).ravel()
        costs.append(COST_FN * fn + COST_FP * fp)
    best_t = float(thresholds[int(np.argmin(costs))])
    yp_opt = (y_proba >= best_t).astype(int)
    from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
    out["cost_threshold"] = {
        "fn_fp_ratio": "5:1",
        "threshold": round(best_t, 3),
        "default_threshold": 0.5,
        "recall_at_opt": round(float(recall_score(y_test, yp_opt)), 3),
        "precision_at_opt": round(float(precision_score(y_test, yp_opt, zero_division=0)), 3),
        "f1_at_opt": round(float(f1_score(y_test, yp_opt)), 3),
        "accuracy_at_opt": round(float(accuracy_score(y_test, yp_opt)), 3),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, costs, color=C["graphite"], lw=2)
    ax.axvline(best_t, color=C["bad"], lw=2, label=f"cost-optimal = {best_t:.2f}")
    ax.axvline(0.5, color=C["mist"], ls="--", lw=1.5, label="default = 0.50")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Total cost (5·FN + 1·FP)")
    ax.set_title("Cost vs threshold (FN 5× costlier than FP)")
    ax.legend(); fig.tight_layout()
    fig.savefig(ART / "fig_threshold.png", dpi=110); plt.close(fig)

    # ----- Save -----
    with open(ART / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Artifacts written to {ART}")
    print(json.dumps({k: out[k] for k in
          ["target", "splits", "cv_final", "test_metrics",
           "split_variance", "cost_threshold"]}, indent=2))


if __name__ == "__main__":
    main()
