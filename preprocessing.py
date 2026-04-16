import numpy as np

# Setting only_cleaveland as true removes ~60% of the data but all features are preserved. 
# Default value =false deletes 3 features with >30% missing data.
def preprocessed_data(df, only_cleveland=False):

    df = df.copy()

    if only_cleveland:
        df = df[df["dataset"] == "Cleveland"].reset_index(drop=True)

        #TODO: normalize columns ca, thal and slope

    else:
        df = df.drop(columns=["ca", "thal", "slope"])

    df = df.drop(columns=["id", "dataset"])

    # Lablel invalid values as missing
    df["trestbps"] = df["trestbps"].replace(0, np.nan)
    df["chol"] = df["chol"].replace(0, np.nan)

    # Replace missing numerical values with median
    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    #TODO: Finish normalization
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})

    return df
