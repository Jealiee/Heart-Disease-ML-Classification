import matplotlib.pyplot as plt
import pandas as pd
import shap

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance")
    plt.show()

def shap_explain(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X)
