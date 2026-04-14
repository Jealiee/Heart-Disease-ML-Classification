import pandas as pd
import numpy as np

#Put the csv file path here (if using different format change pd.read_csv to correct one)
FILE_PATH = 'testdata.csv'

if __name__ == "__main__":
    df = pd.read_csv(FILE_PATH)

    print("\n" + "="*60)    
    print('Shape')
    print(df.shape)

    print("\n" + "="*60)
    print('Info')
    df.info()

    print("\n" + "="*60)
    print('Unique values')
    print(df.nunique())

    print("\n" + "="*60)
    print('Number of missing values in each column')
    print(df.isnull().sum())

    for col in df.columns:
        missing_ratio = df[col].isnull().mean()
        if missing_ratio > 0.3:
            print(f" WARNING! {col} has {missing_ratio:.1%} missing values")

    print("\n" + "="*60)
    print('Percentage of missing values in each column')
    print((df.isnull().sum()/(len(df)))*100)


    cat_cols=df.select_dtypes(include=['object', 'string']).columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    print("\n" + "="*60)
    print("Categorical Variables:")
    print(cat_cols)

    print("\n" + "="*60)
    print("Numerical Variables:")
    print(num_cols)
    
    print("\n" + "="*60)
    print('Statistics summary for numerical datatypes')
    print(df.describe().T)

    print("\n" + "="*60)