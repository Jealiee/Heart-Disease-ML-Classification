import pandas as pd
from data_exploration import explore_data
from pathlib import Path
from preprocessing import clean_data
from data_exploration import plot_distributions


# Put the csv file path here (if using different format change pd.read_csv to correct one)
FILE_PATH = Path("heart_disease_uci.csv")

df = pd.read_csv(FILE_PATH)
target = "num"

if __name__ == "__main__":
    df_full = clean_data(df, only_cleveland=False)
    df_cleveland = clean_data(df, only_cleveland=True)
    ## choose one
    #explore_data(df, target)
    #explore_data(df_full, target)  # raw
    #explore_data(df_cleveland, target)  # filtered
    # run_model(df)
    num_vars = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    plot_distributions(df_full, num_vars)
