import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np

from sklearn.inspection import permutation_importance


def plot_feature_importance(model, feature_names):

    if not hasattr(model, "feature_importances_"):
        print("This model does not have feature_importances_.")
        return

    importances = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.show()


def shap_explain_tree(model, X):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    print(X.columns)
    print("SHAP shape:", np.array(shap_values).shape)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]

    elif len(shap_values.shape) == 3:
        shap_values_to_plot = shap_values[:, :, 1]

    else:
        shap_values_to_plot = shap_values

    print("SHAP to plot shape:", shap_values_to_plot.shape)

    shap.summary_plot(shap_values_to_plot, X, plot_type="dot")

    shap.summary_plot(shap_values_to_plot, X, plot_type="bar")


def plot_logistic_coefficients(model, feature_names):

    coef = model.coef_[0]

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef
    })

    df["abs_coefficient"] = df["coefficient"].abs()

    df = df.sort_values("abs_coefficient", ascending=False)

    plt.figure(figsize=(10, 6))

    plt.barh(df["feature"], df["coefficient"])

    plt.gca().invert_yaxis()

    plt.title("Logistic Regression Coefficients")

    plt.tight_layout()

    plt.show()


def plot_permutation_importance(model, X, y):

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        scoring="f1"
    )

    df = pd.DataFrame({
        "feature": X.columns,
        "importance": result.importances_mean
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))

    plt.barh(df["feature"], df["importance"])

    plt.gca().invert_yaxis()

    plt.title("Permutation Importance")

    plt.tight_layout()

    plt.show()
