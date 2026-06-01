import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_exploration import explore_data
from preprocessing import clean_data, preprocess_fold
from models import logistic_model, decision_tree, random_forest
from evaluation import (
    run_stratified_cv,
    plot_results,
    plot_roc_curves,
    plot_combined_confusion_matrix,
    print_summary_table
)
from explainability import (
    plot_feature_importance,
    shap_explain_tree,
    plot_logistic_coefficients,
    plot_permutation_importance
)


# Put the csv file path here
FILE_PATH = Path("heart_disease_uci.csv")

# Change this manually depending on experiment
RESULTS_FOLDER = Path("results_raw_cut_balanced")
RESULTS_FOLDER.mkdir(exist_ok=True)

df = pd.read_csv(FILE_PATH)
target = "num"


if __name__ == "__main__":

    ######### Data exploration and cleaning ###########

    # IMPORTANT! Make sure only 1 is uncommented before running the script
    df_clean = clean_data(df, only_cleveland=False)
    #df_clean = clean_data(df, only_cleveland=True)

    # explore_data(df, target)
    # explore_data(df_clean, target)

    ######### Get models ###########

    log_model = logistic_model()
    tree_model = decision_tree()
    forest_model = random_forest()

    ######### Run training ###########

    x = df_clean.drop(columns=[target])
    y = df_clean[target]

    n_splits = 5

    results_log = run_stratified_cv(log_model, x, y, n_splits)
    results_tree = run_stratified_cv(tree_model, x, y, n_splits)
    results_forest = run_stratified_cv(forest_model, x, y, n_splits)

    results_all = {
        "Logistic Regression": results_log,
        "Decision Tree": results_tree,
        "Random Forest": results_forest
    }

    print("\nLogistic Regression:", results_log)
    print("\nDecision Tree:", results_tree)
    print("\nRandom Forest:", results_forest)

    ######### Save raw results ###########

    serializable_results = {}

    for model_name, results in results_all.items():
        serializable_results[model_name] = {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in results.items()
        }

    with open(RESULTS_FOLDER / "results.json", "w") as f:
        json.dump(serializable_results, f, indent=4)

    with open(RESULTS_FOLDER / "summary.txt", "w") as f:
        for model_name, results in results_all.items():
            f.write(f"\n{model_name}\n")
            for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
                values = results[metric]
                f.write(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}\n")

    ######### Visualize and save evaluation results ###########

    plot_results(results_all, save_folder=RESULTS_FOLDER)
    print_summary_table(results_all)
    plot_roc_curves(results_all, save_folder=RESULTS_FOLDER)

    plot_combined_confusion_matrix(
        results_log,
        "Logistic Regression",
        save_folder=RESULTS_FOLDER
    )

    plot_combined_confusion_matrix(
        results_tree,
        "Decision Tree",
        save_folder=RESULTS_FOLDER
    )

    plot_combined_confusion_matrix(
        results_forest,
        "Random Forest",
        save_folder=RESULTS_FOLDER
    )
    ######### Explainability / XAI ###########

    x_processed, _ = preprocess_fold(x, x)

    # Logistic Regression explainability
    final_log_model = logistic_model()
    final_log_model.fit(x_processed, y)

    plot_logistic_coefficients(
        final_log_model,
        x_processed.columns,
        save_path=RESULTS_FOLDER / "logistic_coefficients.png"
    )

    plot_permutation_importance(
        final_log_model,
        x_processed,
        y,
        save_path=RESULTS_FOLDER / "logistic_permutation_importance.png"
    )

    # Random Forest explainability
    final_forest_model = random_forest()
    final_forest_model.fit(x_processed, y)

    plot_feature_importance(
        final_forest_model,
        x_processed.columns,
        save_path=RESULTS_FOLDER / "random_forest_feature_importance.png"
    )

    plot_permutation_importance(
        final_forest_model,
        x_processed,
        y,
        save_path=RESULTS_FOLDER / "random_forest_permutation_importance.png"
    )

    # SHAP explanation for Random Forest
    shap_explain_tree(
        final_forest_model,
        x_processed.sample(100, random_state=42),
        save_folder=RESULTS_FOLDER
    )
