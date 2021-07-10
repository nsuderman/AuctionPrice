import os
import pandas as pd


def data_to_df():
    data_location = os.path.abspath(os.getcwd()) + '\data_collection\data\data_pricing_challenge.csv'
    print(f'Data has already been provided - reading data from {data_location}\n')
    return pd.read_csv(data_location)