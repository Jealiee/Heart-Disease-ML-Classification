import numpy as np


# Setting only_cleaveland as true removes ~60% of the data but all features are preserved.
# Default value =false deletes 3 features with >30% missing data.
def clean_data(df, target, only_cleveland=False):

    df = df.copy()

    if only_cleveland:
        df = df[df["dataset"] == "Cleveland"].reset_index(drop=True)

        # TODO: normalize columns ca, thal and slope

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

    # Depending on method may have to move
    # TODO: Handle nominal categorical variables

    return df

def preprocess_fold (X_train, X_val):

    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()

    binary_cols = ["sex", "fbs", "exang"]

    num_cols = [
        col for col in X_train.select_dtypes(include=np.number).columns
        if col not in binary_cols
    ]

    # Imputing missing boolean values with most frequent class
    X_train_processed["fbs"] = X_train["fbs"].fillna(X_train["fbs"].mode()[0])
    X_train_processed["exang"] = X_train["exang"].fillna(X_train["exang"].mode()[0])

    X_val_processed["fbs"] = X_val["fbs"].fillna(X_train["fbs"].mode()[0])
    X_val_processed["exang"] = X_val["exang"].fillna(X_train["exang"].mode()[0])

    # TODO: Scale numerical values

    
    # Replace missing numerical values with median
    for col in num_cols:
        median = X_train[col].median()
        X_train_processed[col] = X_train[col].fillna(median)
        X_val_processed[col] = X_val[col].fillna(median)


    return X_train_processed, X_val_processed
