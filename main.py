import pandas as pd
from data_exploration import explore_data
from pathlib import Path
from preprocessing import preprocessed_data
from model import run_model

# Put the csv file path here (if using different format change pd.read_csv to correct one)
FILE_PATH = Path("heart_disease_uci.csv")

df = pd.read_csv(FILE_PATH)
target = "num"

if __name__ == "__main__":
    # df_full = preprocessed_data(df)
    # df_cleveland = preprocessed_data(df, only_cleveland=True)
    ## choose one
    explore_data(df, target)
    # explore_data(df_full, target)  # raw
    # explore_data(df_cleveland, target)  # filtered
    # run_model(df)
