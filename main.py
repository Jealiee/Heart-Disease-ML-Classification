from pathlib import Path
import pandas as pd

from preprocessing import clean_data
from models import logistic_model, decision_tree, random_forest, PARAM_GRIDS
from evaluation import (
    tune_model_grid,
    train_and_evaluate_final,
    compare_models_statistically,
    plot_cv_comparison,
    save_json,
)
from explainability import (
    plot_model_importance,
    plot_permutation_importance,
    plot_shap_explanations,
    plot_local_explanation,
    plot_error_case_explanations,
)

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "heart_disease_uci.csv"
TARGET = "num"
OUTPUT_DIR = BASE_DIR / "outputs"

DATASET_VARIANTS = {
    "full_cut_features": False,
    "cleveland_all_features": True,
}
BALANCING_VARIANTS = {
    "unbalanced": None,
    "balanced": "balanced",
}


def build_models(class_weight):
    return {
        "Logistic Regression": logistic_model(class_weight=class_weight),
        "Decision Tree": decision_tree(class_weight=class_weight),
        "Random Forest": random_forest(class_weight=class_weight),
    }


def run_experiment(df, dataset_name, only_cleveland, balancing_name, class_weight):
    experiment_name = f"{dataset_name}_{balancing_name}"
    run_dir = OUTPUT_DIR / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    df_clean = clean_data(df, only_cleveland=only_cleveland)
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]

    n_splits = 5
    scoring = "f1"
    model_builders = build_models(class_weight=class_weight)

    searches = {}
    cv_results = {}

    print(f" EXPERIMENT: {experiment_name}", flush=True)
    print(f"Samples: {len(df_clean)}, Features before encoding: {X.shape[1]}", flush=True)

    for model_name, model in model_builders.items():
        print(f"Tuning {model_name}...", flush=True)
        search = tune_model_grid(
            model=model,
            param_grid=PARAM_GRIDS[model_name],
            X=X,
            y=y,
            n_splits=n_splits,
            scoring=scoring,
        )
        searches[model_name] = search
        cv_results[model_name] = search["best_cv_scores"]
        search["search_results"].to_csv(
            run_dir / f"{model_name.lower().replace(' ', '_')}_grid_search_results.csv",
            index=False,
        )
        print(f"  best {scoring}: {search['best_score']:.4f}; params: {search['best_params']}", flush=True)

    cv_summary = plot_cv_comparison(cv_results, run_dir)
    cv_summary.insert(0, "experiment", experiment_name)
    cv_summary.to_csv(run_dir / "cv_summary.csv", index=False)

    stats_lr_rf = compare_models_statistically(
        cv_results["Logistic Regression"],
        cv_results["Random Forest"],
        "Logistic Regression",
        "Random Forest",
        metric=scoring,
    )
    save_json(stats_lr_rf, run_dir / "statistical_test_logistic_vs_random_forest.json")

    best_model_name = max(searches, key=lambda name: searches[name]["best_score"])
    best_model = searches[best_model_name]["best_model"]
    best_params = searches[best_model_name]["best_params"]

    final = train_and_evaluate_final(best_model, X, y, run_dir, n_splits=5)
    save_json(final["metrics"], run_dir / "final_cv_metrics.json")

  
    plot_model_importance(final["model"], final["X_test_processed"].columns, run_dir, best_model_name)
    plot_permutation_importance(
        final["model"], final["X_test_processed"], final["y_test"], run_dir, best_model_name, scoring=scoring, n_repeats=5
    )
    plot_local_explanation(
        final["model"], final["X_test_processed"], final["y_test"], final["y_pred"], run_dir, best_model_name
    )
    plot_error_case_explanations(
        final["model"],
        final["X_test_processed"],
        final["y_test"],
        final["y_pred"],
        run_dir,
        best_model_name
    )

    experiment_summary = {
        "experiment": experiment_name,
        "dataset_name": dataset_name,
        "balancing": balancing_name,
        "n_samples": int(len(df_clean)),
        "n_features_before_encoding": int(X.shape[1]),
        "class_0_fraction": float((y == 0).mean()),
        "class_1_fraction": float((y == 1).mean()),
        "selected_model": best_model_name,
        "best_params": best_params,
        "best_cv_f1": float(searches[best_model_name]["best_score"]),
        **{f"final_cv_{k}": float(v) for k, v in final["metrics"].items()},
    }
    save_json(experiment_summary, run_dir / "experiment_summary.json")
    print(f"Selected final model: {best_model_name}; final CV F1={final['metrics']['f1']:.4f}", flush=True)

    explainability_payload = {
        "model": final["model"],
        "X_test_processed": final["X_test_processed"],
        "run_dir": run_dir,
        "model_name": best_model_name,
    }
    return experiment_summary, explainability_payload, cv_results


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(FILE_PATH)
    all_summaries = []
    explanation_payloads = {}
    all_cv_results = {}

    for dataset_name, only_cleveland in DATASET_VARIANTS.items():
        for balancing_name, class_weight in BALANCING_VARIANTS.items():
            summary, payload, cv_results = run_experiment(df, dataset_name, only_cleveland, balancing_name, class_weight)
            all_summaries.append(summary)
            explanation_payloads[summary["experiment"]] = payload
            all_cv_results[summary["experiment"]] = cv_results

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTPUT_DIR / "all_experiments_summary.csv", index=False)

    extra_tests = []

    
    extra_tests.append(
        compare_models_statistically(
            all_cv_results["full_cut_features_unbalanced"]["Random Forest"],
            all_cv_results["full_cut_features_balanced"]["Random Forest"],
            "Full_unbalanced_RF",
            "Full_balanced_RF",
            metric="f1",
        )
    )

   
    extra_tests.append(
        compare_models_statistically(
            all_cv_results["cleveland_all_features_unbalanced"]["Logistic Regression"],
            all_cv_results["cleveland_all_features_balanced"]["Logistic Regression"],
            "Cleveland_unbalanced_LR",
            "Cleveland_balanced_LR",
            metric="f1",
        )
    )


    extra_tests.append(
        compare_models_statistically(
            all_cv_results["cleveland_all_features_unbalanced"]["Logistic Regression"],
            all_cv_results["full_cut_features_unbalanced"]["Random Forest"],
            "Cleveland_unbalanced_LR",
            "Full_unbalanced_RF",
            metric="f1",
        )
    )

    extra_tests_df = pd.DataFrame(extra_tests)
    extra_tests_df.to_csv(OUTPUT_DIR / "additional_statistical_tests.csv", index=False)
    best_row = summary_df.sort_values("final_cv_f1", ascending=False).iloc[0]
    best_payload = explanation_payloads[best_row["experiment"]]
    X_shap = best_payload["X_test_processed"].sample(
        min(100, len(best_payload["X_test_processed"])), random_state=42
    )
    plot_shap_explanations(
        best_payload["model"], X_shap, best_payload["run_dir"], best_payload["model_name"]
    )

    print("ALL EXPERIMENTS SUMMARY")
    print(summary_df)
    print(f"\nBest overall experiment for report figures: {best_row['experiment']}")
    print(f"All outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
