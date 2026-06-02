from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def logistic_model(class_weight=None):
    return LogisticRegression(
        max_iter=5000,
        solver="liblinear",
        random_state=42,
        class_weight=class_weight,
    )


def decision_tree(class_weight=None):
    return DecisionTreeClassifier(
        random_state=42,
        class_weight=class_weight,
    )


def random_forest(class_weight=None):
    return RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight,
    )


PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
    },
    "Decision Tree": {
        "max_depth": [None, 3, 5, 8],
        "min_samples_leaf": [1, 2, 4],
    },
    "Random Forest": {
        "n_estimators": [100],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt"],
    },
}
