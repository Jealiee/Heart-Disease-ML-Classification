from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from scipy.stats import wilcoxon, ttest_rel

from preprocessing import preprocess_fold


METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _predict_scores(model, X):
   
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def compute_metrics(y_true, y_pred, y_score=None):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        results["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        results["roc_auc"] = np.nan
    return results


def run_stratified_cv(model, X, y, n_splits=5):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {m: [] for m in METRICS}

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_p, X_val_p = preprocess_fold(X_train, X_val)

        model_clone = clone(model)
        model_clone.fit(X_train_p, y_train)

        y_pred = model_clone.predict(X_val_p)
        y_score = _predict_scores(model_clone, X_val_p)
        fold_scores = compute_metrics(y_val, y_pred, y_score)

        for metric in METRICS:
            scores[metric].append(fold_scores[metric])

    return scores


def tune_model_grid(model, param_grid, X, y, n_splits=5, scoring="f1"):
   
    rows = []
    best_score = -np.inf
    best_params = None
    best_cv_scores = None

    for params in ParameterGrid(param_grid):
        candidate = clone(model).set_params(**params)
        cv_scores = run_stratified_cv(candidate, X, y, n_splits=n_splits)
        mean_score = float(np.mean(cv_scores[scoring]))
        std_score = float(np.std(cv_scores[scoring]))

        row = {"params": params, f"mean_{scoring}": mean_score, f"std_{scoring}": std_score}
        for metric in METRICS:
            row[f"mean_{metric}"] = float(np.mean(cv_scores[metric]))
            row[f"std_{metric}"] = float(np.std(cv_scores[metric]))
        rows.append(row)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_cv_scores = cv_scores

    search_results = pd.DataFrame(rows).sort_values(f"mean_{scoring}", ascending=False)
    best_model = clone(model).set_params(**best_params)

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_score": best_score,
        "best_cv_scores": best_cv_scores,
        "search_results": search_results,
    }


def train_and_evaluate_final(model, X, y, output_dir, n_splits=5):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    all_y_score = []
    fold_metrics = []

    last_model = None
    last_X_val_p = None
    last_y_val = None
    last_y_pred = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_p, X_val_p = preprocess_fold(X_train, X_val)

        model_fold = clone(model)
        model_fold.fit(X_train_p, y_train)

        y_pred = model_fold.predict(X_val_p)
        y_score = _predict_scores(model_fold, X_val_p)

        fold_score = compute_metrics(y_val, y_pred, y_score)
        fold_score["fold"] = fold
        fold_metrics.append(fold_score)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_score.extend(y_score)

        last_model = model_fold
        last_X_val_p = X_val_p
        last_y_val = y_val
        last_y_pred = y_pred

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_score = np.array(all_y_score)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(output_dir / "final_cv_fold_metrics.csv", index=False)

    metrics = {}
    for metric in METRICS:
        metrics[metric] = float(fold_metrics_df[metric].mean())
        metrics[f"{metric}_std"] = float(fold_metrics_df[metric].std())

    cm = confusion_matrix(all_y_true, all_y_pred)
    report = classification_report(all_y_true, all_y_pred, zero_division=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        cm,
        display_labels=["No disease", "Disease"]
    ).plot(ax=ax, values_format="d")
    ax.set_title("Final 5-Fold CV Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_final_cv.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(all_y_true, all_y_score, ax=ax)
    ax.set_title("Final 5-Fold CV ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve_final_cv.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(all_y_true, all_y_score, ax=ax)
    ax.set_title("Final 5-Fold CV Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve_final_cv.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "model": last_model,
        "X_test_processed": last_X_val_p,
        "y_test": last_y_val,
        "y_pred": last_y_pred,
        "metrics": metrics,
        "fold_metrics": fold_metrics_df,
        "confusion_matrix": cm,
        "classification_report": report,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
        "all_y_score": all_y_score,
    }
def compare_models_statistically(cv_scores_a, cv_scores_b, model_a, model_b, metric="f1"):
    
    a = np.array(cv_scores_a[metric], dtype=float)
    b = np.array(cv_scores_b[metric], dtype=float)

    result = {
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        f"{model_a}_mean": float(np.mean(a)),
        f"{model_b}_mean": float(np.mean(b)),
    }

    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        result["wilcoxon_statistic"] = float(stat)
        result["wilcoxon_p_value"] = float(p)
    except ValueError:
        result["wilcoxon_statistic"] = None
        result["wilcoxon_p_value"] = None

    stat_t, p_t = ttest_rel(a, b)
    result["paired_t_statistic"] = float(stat_t)
    result["paired_t_p_value"] = float(p_t)
    return result


def plot_cv_comparison(results_dict, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_name, scores in results_dict.items():
        row = {"model": model_name}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(np.mean(scores[metric]))
            row[f"{metric}_std"] = float(np.std(scores[metric]))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "cv_summary.csv", index=False)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(results_dict.keys())
        values = [results_dict[name][metric] for name in labels]
        ax.boxplot(values, labels=labels)
        ax.set_title(f"Cross-validated {metric} across models")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / f"cv_{metric}_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return summary


def save_json(data, path):
    def default(o):
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default)
