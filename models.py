from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# def logistic_model(max_iter):
#     return LogisticRegression(
#         max_iter=max_iter,
#         solver="lbfgs",
#         random_state=42,
#         class_weight="balanced"
#     )
#
#
# def decision_tree(max_depth):
#     return DecisionTreeClassifier(
#         max_depth=max_depth,
#         random_state=42,
#         class_weight="balanced"
#     )
#
#
# def random_forest(max_depth):
#     return RandomForestClassifier(
#         n_estimators=100,
#         max_depth=max_depth,
#         random_state=42,
#         n_jobs=-1,
#         class_weight="balanced"
#     )

def logistic_model():
    return LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="liblinear",
        random_state=42
    )

def decision_tree():
    return DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        random_state=42
    )

def random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=42
    )
