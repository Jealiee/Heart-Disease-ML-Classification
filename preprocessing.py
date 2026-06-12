import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# Setting only_cleveland as True removes ~60% of the data but all features are preserved.
# Default value False keeps all records but removes 3 features with >30% missing data.
def clean_data(df, only_cleveland=False):
    df = df.copy()
    df = df.drop_duplicates()

    if only_cleveland:
        df = df[df["dataset"] == "Cleveland"].reset_index(drop=True)
    else:
        df = df.drop(columns=["ca", "thal", "slope"])

    df = df.drop(columns=["id", "dataset"])

    # Label invalid values as missing
    df["trestbps"] = df["trestbps"].replace(0, np.nan)
    df["chol"] = df["chol"].replace(0, np.nan)
    df["oldpeak"] = df["oldpeak"].clip(lower=0)

    # Encoding binary values
    df["sex"] = (
        df["sex"].astype(str).str.strip().str.lower().map({"male": 1, "female": 0})
    )

    df["fbs"] = (
        df["fbs"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    )

    df["exang"] = (
        df["exang"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    )

    # Convert multiclass target into binary target
    df["num"] = (df["num"] > 0).astype(int)

    return df


def preprocess_fold(X_train, X_val):
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()

    num_cols = [
        c
        for c in ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
        if c in X_train_processed.columns
    ]

    cat_cols = ["cp", "restecg", "slope", "thal"]
    cat_cols = [c for c in cat_cols if c in X_train_processed.columns]

    # Impute missing binary values with most frequent class from training fold only
    for col in ["fbs", "exang"]:
        if col in X_train_processed.columns:
            mode_value = X_train_processed[col].mode()[0]
            X_train_processed[col] = X_train_processed[col].fillna(mode_value)
            X_val_processed[col] = X_val_processed[col].fillna(mode_value)

    # One-hot encode categorical variables using training fold only
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        train_encoded = encoder.fit_transform(X_train_processed[cat_cols])
        val_encoded = encoder.transform(X_val_processed[cat_cols])

        train_encoded = pd.DataFrame(
            train_encoded,
            columns=encoder.get_feature_names_out(cat_cols),
            index=X_train_processed.index,
        )

        val_encoded = pd.DataFrame(
            val_encoded,
            columns=encoder.get_feature_names_out(cat_cols),
            index=X_val_processed.index,
        )

        X_train_processed = X_train_processed.drop(columns=cat_cols)
        X_val_processed = X_val_processed.drop(columns=cat_cols)

        X_train_processed = pd.concat([X_train_processed, train_encoded], axis=1)
        X_val_processed = pd.concat([X_val_processed, val_encoded], axis=1)

    # Median imputation for numerical variables using training fold only
    for col in num_cols:
        median = X_train_processed[col].median()
        X_train_processed[col] = X_train_processed[col].fillna(median)
        X_val_processed[col] = X_val_processed[col].fillna(median)

    # Winsorization using training fold limits only
    cols_to_fix = ["trestbps", "chol"]

    for col in cols_to_fix:
        if col not in X_train_processed.columns:
            continue

        q1 = X_train_processed[col].quantile(0.25)
        q3 = X_train_processed[col].quantile(0.75)
        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        X_train_processed[col] = np.where(
            X_train_processed[col] > upper_limit,
            upper_limit,
            np.where(
                X_train_processed[col] < lower_limit,
                lower_limit,
                X_train_processed[col],
            ),
        )

        X_val_processed[col] = np.where(
            X_val_processed[col] > upper_limit,
            upper_limit,
            np.where(
                X_val_processed[col] < lower_limit,
                lower_limit,
                X_val_processed[col],
            ),
        )

    # Scale numerical values using training fold only
    scaler = RobustScaler()
    X_train_processed[num_cols] = scaler.fit_transform(X_train_processed[num_cols])
    X_val_processed[num_cols] = scaler.transform(X_val_processed[num_cols])

    return X_train_processed, X_val_processed
