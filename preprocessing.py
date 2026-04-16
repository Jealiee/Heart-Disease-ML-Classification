import numpy as np


# Setting only_cleaveland as true removes ~60% of the data but all features are preserved.
# Default value =false deletes 3 features with >30% missing data.
def preprocessed_data(df, target, only_cleveland=False):

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

    # !!! MOVE TO SEP FUNCTION - DATA LEAKAGE RISK
    # Replace missing numerical values with median
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

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

    # !!! MOVE TO SEP FUNCTION - DATA LEAKAGE RISK
    # Imputing missing boolean values with most frequent class
    df["fbs"] = df["fbs"].fillna(df["fbs"].mode()[0])
    df["exang"] = df["exang"].fillna(df["exang"].mode()[0])

    # Depending on method may have to move
    # TODO: Handle nominal categorical variables

    # !!! MOVE TO SEP FUNCTION - DATA LEAKAGE RISK
    # TODO: Scale numerical values

    return df
