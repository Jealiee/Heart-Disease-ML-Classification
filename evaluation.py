from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

    for train_idx, val_idx in skf.split(x, y):
        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        x_train_p, x_val_p = preprocess_fold(x_train, x_val)

        model_clone = clone(model)
        model_clone.fit(x_train_p, y_train)

        y_pred = model_clone.predict(x_val_p)

        acc_scores.append(accuracy_score(y_val, y_pred))
        prec_scores.append(precision_score(y_val, y_pred, average="macro"))
        rec_scores.append(recall_score(y_val, y_pred, average="macro"))
        f1_scores.append(f1_score(y_val, y_pred, average="macro"))

    return {
        "accuracy": acc_scores,
        "precision": prec_scores,
        "recall": rec_scores,
        "f1": f1_scores,
    }

def plot_results(results_dict):

    metrics = ["accuracy", "precision", "recall", "f1"]

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

        bars = plt.bar(x, means, yerr=stds, capsize=5)

        for i, v in enumerate(means):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

        plt.xticks(x, model_names, rotation=20)
        plt.title(f"{metric.capitalize()} Comparison")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()
