import pandas as pd
from data_exploration import explore_data
from pathlib import Path
from preprocessing import preprocessed_data
from model import run_model

#Put the csv file path here (if using different format change pd.read_csv to correct one)
FILE_PATH = Path("heart_disease_uci.csv")

df = pd.read_csv(FILE_PATH)

if __name__ == "__main__":
    preprocessed_data(df)
    explore_data(df, target ='num')
    run_model(df)

