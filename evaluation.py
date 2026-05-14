from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.base import clone
from preprocessing import preprocess_fold

import matplotlib.pyplot as plt
import numpy as np


def run_stratified_cv(model, x, y, n_splits=5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    auc_scores = []

    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        x_train_p, x_val_p = preprocess_fold(x_train, x_val)

        model_clone = clone(model)
        model_clone.fit(x_train_p, y_train)

        y_pred = model_clone.predict(x_val_p)

        if hasattr(model_clone, "predict_proba"):
            y_proba = model_clone.predict_proba(x_val_p)[:, 1]
        else:
            y_proba = model_clone.decision_function(x_val_p)

        cm = confusion_matrix(y_val, y_pred)
        print(f"\nFold {fold} confusion matrix:")
        print(cm)

        acc_scores.append(accuracy_score(y_val, y_pred))
        prec_scores.append(precision_score(y_val, y_pred, average="macro"))
        rec_scores.append(recall_score(y_val, y_pred, average="macro"))
        f1_scores.append(f1_score(y_val, y_pred, average="macro"))
        auc_scores.append(roc_auc_score(y_val, y_proba))

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

    return {
        "accuracy": acc_scores,
        "precision": prec_scores,
        "recall": rec_scores,
        "f1": f1_scores,
        "auc": auc_scores,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "y_proba": np.array(all_y_proba),
    }


def plot_results(results_dict, save_folder=None):

    metrics = ["accuracy", "precision", "recall", "f1", "auc"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))

        model_names = []
        means = []
        stds = []

        for model_name, results in results_dict.items():
            values = results[metric]
            model_names.append(model_name)
            means.append(np.mean(values))
            stds.append(np.std(values))

        x = np.arange(len(model_names))

        plt.bar(x, means, yerr=stds, capsize=5)

        for i, v in enumerate(means):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

        plt.xticks(x, model_names, rotation=20)
        plt.title(f"{metric.upper()} Comparison")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.tight_layout()
        if save_folder is not None:
           plt.savefig(save_folder / f"{metric}_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_roc_curves(results_dict, save_folder=None):

    plt.figure(figsize=(8, 6))

    for model_name, results in results_dict.items():
        fpr, tpr, _ = roc_curve(results["y_true"], results["y_proba"])
        auc_value = roc_auc_score(results["y_true"], results["y_proba"])

        plt.plot(fpr, tpr, label=f"{model_name} AUC = {auc_value:.2f}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    if save_folder is not None:
       plt.savefig(save_folder / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_combined_confusion_matrix(results, model_name, save_folder=None):

    cm = confusion_matrix(results["y_true"], results["y_pred"])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Healthy", "Heart disease"]
    )

    disp.plot()
    plt.title(f"{model_name} Combined Confusion Matrix")
    plt.tight_layout()
    if save_folder is not None:
       filename = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
       plt.savefig(save_folder / filename, dpi=300, bbox_inches="tight")
    plt.show()


def print_summary_table(results_dict):

    print("\nFinal summary: mean ± std")
    print("-" * 80)

    metrics = ["accuracy", "precision", "recall", "f1", "auc"]

    for model_name, results in results_dict.items():
        print(f"\n{model_name}")
        for metric in metrics:
            values = results[metric]
            print(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")
