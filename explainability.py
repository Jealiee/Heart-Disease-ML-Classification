from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def _save_barh(df, value_col, title, filename, top_n=15):
    df_plot = df.head(top_n).sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df_plot["feature"], df_plot[value_col])
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_importance(model, feature_names, output_dir, model_name="model", top_n=15):
    """Native importance for tree models or coefficients for logistic regression."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        value_name = "importance"
        title = f"{model_name}: native feature importance"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
        value_name = "abs_coefficient"
        title = f"{model_name}: absolute logistic coefficients"
    else:
        return None

    df = pd.DataFrame({"feature": feature_names, value_name: values})
    df = df.sort_values(value_name, ascending=False)
    csv_path = output_dir / f"{model_name.lower().replace(' ', '_')}_native_importance.csv"
    png_path = output_dir / f"{model_name.lower().replace(' ', '_')}_native_importance.png"
    df.to_csv(csv_path, index=False)
    _save_barh(df.rename(columns={value_name: "value"}), "value", title, png_path, top_n=top_n)
    return df


def plot_permutation_importance(model, X_test, y_test, output_dir, model_name="model", scoring="f1", top_n=15, n_repeats=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring=scoring, n_jobs=1
    )
    df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    csv_path = output_dir / f"{model_name.lower().replace(' ', '_')}_permutation_importance.csv"
    png_path = output_dir / f"{model_name.lower().replace(' ', '_')}_permutation_importance.png"
    df.to_csv(csv_path, index=False)
    _save_barh(df.rename(columns={"importance_mean": "value"}), "value", f"{model_name}: permutation importance", png_path, top_n=top_n)
    return df


def plot_shap_explanations(model, X_sample, output_dir, model_name="model", max_display=15):
    """Create SHAP global summary and one local waterfall plot when SHAP works."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap
    except ImportError:
        print("SHAP is not installed, skipping SHAP plots.")
        return None

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_to_plot = shap_values[1]
                base_value = explainer.expected_value[1]
            elif getattr(shap_values, "ndim", 0) == 3:
                shap_to_plot = shap_values[:, :, 1]
                base_value = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value
            else:
                shap_to_plot = shap_values
                base_value = explainer.expected_value

            plt.figure()
            shap.summary_plot(shap_to_plot, X_sample, show=False, max_display=max_display)
            plt.tight_layout()
            plt.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_shap_summary.png", dpi=300, bbox_inches="tight")
            plt.close()

            mean_abs = np.abs(shap_to_plot).mean(axis=0)
            shap_df = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
            shap_df.to_csv(output_dir / f"{model_name.lower().replace(' ', '_')}_mean_abs_shap.csv", index=False)

            # Local SHAP explanation for the first test sample.
            try:
                explanation = shap.Explanation(
                    values=shap_to_plot[0],
                    base_values=base_value,
                    data=X_sample.iloc[0].values,
                    feature_names=list(X_sample.columns),
                )
                shap.plots.waterfall(explanation, max_display=max_display, show=False)
                plt.tight_layout()
                plt.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_shap_local_waterfall.png", dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"Local SHAP waterfall failed for {model_name}: {e}")

            return shap_df

        print(f"SHAP skipped for {model_name}: non-tree model. Use permutation/coefficient plots instead.")
        return None
    except Exception as e:
        print(f"SHAP failed for {model_name}: {e}")
        return None

def plot_local_linear_contribution(model, X_test, y_test, y_pred, output_dir, model_name="model", sample_position=0, top_n=12):
    """Local explanation for linear/logistic models: coefficient * feature value contributions."""
    if not hasattr(model, "coef_"):
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = X_test.iloc[sample_position]
    contributions = model.coef_.ravel() * x.values
    df = pd.DataFrame({
        "feature": X_test.columns,
        "feature_value": x.values,
        "contribution": contributions,
        "abs_contribution": np.abs(contributions),
    }).sort_values("abs_contribution", ascending=False)

    df.to_csv(output_dir / f"{model_name.lower().replace(' ', '_')}_local_explanation_sample.csv", index=False)

    df_plot = df.head(top_n).sort_values("contribution")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df_plot["feature"], df_plot["contribution"])
    ax.axvline(0, linewidth=1)
    ax.set_title(
        f"{model_name}: local contribution explanation\n"
        f"True={y_test.iloc[sample_position]}, Predicted={y_pred[sample_position]}"
    )
    ax.set_xlabel("coefficient × processed feature value")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_local_explanation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return df


def plot_local_explanation(model, X_test, y_test, y_pred, output_dir, model_name="model", sample_position=0, top_n=12):
    """Create a local explanation for the selected final model.
    For Logistic Regression it uses coefficient × value contributions.
    For tree models the local SHAP waterfall is attempted in plot_shap_explanations.
    """
    if hasattr(model, "coef_"):
        return plot_local_linear_contribution(
            model, X_test, y_test, y_pred, output_dir, model_name, sample_position, top_n
        )
    return None
