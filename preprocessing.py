import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import RobustScaler


# Setting only_cleaveland as true removes ~60% of the data but all features are preserved.
# Default value =false deletes 3 features with >30% missing data.
def clean_data(df,only_cleveland=False):

    df = df.copy()
    df = df.drop_duplicates()
    if only_cleveland:
        df = df[df["dataset"] == "Cleveland"].reset_index(drop=True)

    else:
        df = df.drop(columns=["ca", "thal", "slope"])

    df = df.drop(columns=["id", "dataset"])

    # Lablel invalid values as missing
    df["trestbps"] = df["trestbps"].replace(0, np.nan)
    df["chol"] = df["chol"].replace(0, np.nan)

    # Encoding boolean / binary values
    df["sex"] = (
        df["sex"].astype(str).str.strip().str.lower().map({"male": 1, "female": 0})
    )
    df["fbs"] = (
        df["fbs"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    )
    df["exang"] = (
        df["exang"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    )
    cols_to_fix = ['trestbps', 'chol', 'oldpeak']

    for col in cols_to_fix:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr

        # Cap the values
        df[col] = np.where(df[col] > upper_limit, upper_limit,
                           np.where(df[col] < lower_limit, lower_limit, df[col]))

    # Fill missing numerical values with the median
    numerical_cols = ['trestbps', 'chol', 'thalch', 'oldpeak']
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

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

    # Imputing missing boolean values with most frequent class
    fbs_mode = X_train["fbs"].mode()[0]
    exang_mode = X_train["exang"].mode()[0]

    X_train_processed["fbs"] = X_train_processed["fbs"].fillna(fbs_mode)
    X_train_processed["exang"] = X_train_processed["exang"].fillna(exang_mode)

    X_val_processed["fbs"] = X_val_processed["fbs"].fillna(fbs_mode)
    X_val_processed["exang"] = X_val_processed["exang"].fillna(exang_mode)

    # Encode nominal categorical variables
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

    # Replace missing numerical values with median
    for col in num_cols:
        median = X_train_processed[col].median()
        X_train_processed[col] = X_train_processed[col].fillna(median)
        X_val_processed[col] = X_val_processed[col].fillna(median)

    # TODO: Scale numerical values
    scaler = RobustScaler()
    X_train_processed[num_cols] = scaler.fit_transform(X_train_processed[num_cols])
    X_val_processed[num_cols] = scaler.transform(X_val_processed[num_cols])

    return X_train_processed, X_val_processed
