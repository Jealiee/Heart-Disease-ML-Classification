import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def explore_data(df, target):
    # Basic info
    print("\n" + "=" * 60)
    print("Shape")
    print(df.shape)

    print("\n" + "=" * 60)
    print("Info")
    df.info()

    # Unique, missing and duplicate values
    print("\n" + "=" * 60)
    print("Unique values")
    print(df.nunique())

    print("\n" + "=" * 60)
    print("Number of missing values in each column")
    print(df.isnull().sum())

    for col in df.columns:
        missing_ratio = df[col].isnull().mean()
        if missing_ratio > 0.3:
            print(f" WARNING! {col} has {missing_ratio:.1%} missing values")

    print("\n" + "=" * 60)
    print("Percentage of missing values in each column")
    print((df.isnull().sum() / (len(df))) * 100)
    
    total_missing = df.isnull().sum().sum()
    total_values = df.shape[0] * df.shape[1]
    overall_missing_pct = (total_missing / total_values) * 100

    print("\n" + "=" * 60)
    print(f"Overall missing values: {overall_missing_pct:.2f}%")

    print("\n" + "=" * 60)
    print("Duplicate rows:")
    print(df.duplicated().sum())

    # Showing which columns are categorical / numerical

    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    print("\n" + "=" * 60)
    print("Categorical Variables:")
    print(cat_cols)

    print("\n" + "=" * 60)
    print("Numerical Variables:")
    print(num_cols)

    print("\n" + "=" * 60)
    print("Statistics summary for numerical datatypes")
    print(df.describe().T)

    # Target distribution
    print("\n" + "=" * 60)
    print("Target distribution:")
    print(df[target].value_counts(normalize=True) * 100)
    
    sns.countplot(x=df[target])
    plt.title("Target Distribution")
    plt.show()

    # Correlation
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[num_cols].corr(method="spearman"), annot=True, cmap="coolwarm")
    plt.title("Spearman Correlation Matrix")
    plt.show()

    # Check data distribution for numerical values
    for col in num_cols:
        if col == 'num':
            continue
        plt.figure()
        df[col].hist(bins=30)
        plt.title(col)
        plt.show()

    # Outliers
    n_cols = len(num_cols)
    fig, axes = plt.subplots(n_cols, 2, figsize=(12, n_cols * 4), squeeze=False)

    for i, col in enumerate(num_cols):
        # Histogram with KDE (Center and Shape)
        sns.histplot(df[col], kde=True, ax=axes[i, 0], color="skyblue")
        axes[i, 0].set_title(f"Distribution of {col}")

        # Boxplot (Spread and Outliers)
        sns.boxplot(x=df[col], ax=axes[i, 1], color="salmon")
        axes[i, 1].set_title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv("heart_disease_uci.csv")
    explore_data(df, 'num')