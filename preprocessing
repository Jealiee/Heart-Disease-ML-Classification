import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocessed_data(df):
    # Drop useless or high-missing columns
    df = df.drop(columns=['id', 'ca', 'thal', 'slope'])
    df = df.dropna(subset=['fbs', 'exang'])

    # Fix impossible values
    df['trestbps'] = df['trestbps'].replace(0, np.nan)
    df['chol'] = df['chol'].replace(0, np.nan)

    # Fill numerical columns with MEDIAN
    num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # coding (features to 0/1)
    df['sex'] = df['sex'].map({'M': 1, 'F': 0})

    # 4. Fill categorical columns with MODE - that is a maybe can mess with the results and data
    # cat_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    # for col in cat_cols:
        # df[col].fillna(df[col].mode()[0], inplace=True)

    return df
