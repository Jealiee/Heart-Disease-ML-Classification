import pandas as pd
from data_exploration import explore_data
from pathlib import Path

#Put the csv file path here (if using different format change pd.read_csv to correct one)
FILE_PATH = Path("heart_disease_uci.csv")

df = pd.read_csv(FILE_PATH)

if __name__ == "__main__":
    explore_data(df, target ='num')

