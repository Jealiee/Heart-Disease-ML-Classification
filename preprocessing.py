import numpy as np
def preprocessed_data(df, only_cleveland=False):

    df = df.copy()

    # Optional filtering
    if only_cleveland:
        df = df[df['dataset'] == 'Cleveland'].reset_index(drop=True)
        # drop only id
        df = df.drop(columns=['id'])

        # encode sex
        df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

        return df

    # Drop columns
    df = df.drop(columns=['id', 'ca', 'thal', 'slope'])

    # Fix values
    df['trestbps'] = df['trestbps'].replace(0, np.nan)
    df['chol'] = df['chol'].replace(0, np.nan)

    # Fill numerical
    num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Encode
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

    return df
