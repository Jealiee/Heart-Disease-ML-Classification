import pandas as pd
import shap
from data_exploration import explore_data
from pathlib import Path
from preprocessing import clean_data
from models import logistic_model, decision_tree, random_forest
from evaluation import run_stratified_cv
from evaluation import plot_results
# from explainability import plot_feature_importance, shap_explain
# from sklearn.ensemble import RandomForestClassifier
import sys
print(sys.executable)
# Put the csv file path here (if using different format change pd.read_csv to correct one)
FILE_PATH = Path("heart_disease_uci.csv")

df = pd.read_csv(FILE_PATH)
target = "num"

if __name__ == "__main__":
    ######### Data exploration and cleaning ###########

    # IMPORTANT! Make sure only 1 is unccommented before running the script
    #df_clean = clean_data(df, only_cleveland=False)
    df_clean = clean_data(df, only_cleveland=True)

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

    print("\nLogistic Regression:", results_log)
    print("\nDecision Tree:", results_tree)
    print("\nRandom Forest:", results_forest)

    ######### TODO: Visualize results ###########

    plot_results({
        "Logistic Regression": results_log,
        "Decision Tree": results_tree,
        "Random Forest": results_forest
    })

    # final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    # final_model.fit(x, y)
    #
    # plot_feature_importance(final_model, x.columns)
    # shap_explain(final_model, x.sample(100))
