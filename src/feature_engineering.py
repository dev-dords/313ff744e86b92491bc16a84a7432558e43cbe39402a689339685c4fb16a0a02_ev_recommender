import os
import re
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from data_preprocessing import load_data

src_dir = os.getcwd()
data_dir = os.path.join(src_dir, 'data')
silver_dir = os.path.join(data_dir, 'silver')
gold_dir = os.path.join(data_dir, 'gold')

def one_hot_encode_data(ev_data):
    """
    One-hot encode categorical columns in the DataFrame.

    Parameters:
    ev_data (pandas.DataFrame): DataFrame containing electric vehicle data.

    Returns:
    pandas.DataFrame: DataFrame with one-hot encoded categorical columns.
    """
    encoder = OneHotEncoder(sparse_output=False)
    categorical_columns = ['brand', 'fast_charge_port', 'segment', 'drivetrain', 'car_body_type']

    ev_encoded = encoder.fit_transform(ev_data[categorical_columns])
    columns = encoder.get_feature_names_out(categorical_columns)

    ev_one_hot_df = pd.DataFrame(ev_encoded, columns=columns)
    ev_encoded_df = pd.concat([ev_data.drop(columns=categorical_columns), ev_one_hot_df], axis=1)

    return ev_encoded_df

def main():
  print("Starting one-hot encoding of electric vehicle data...")
  pre_processed_data = load_data(os.path.join(silver_dir, 'electric_vehicles_preprocessed.csv'))
  ev_encoded_df = one_hot_encode_data(pre_processed_data)
  print("One-hot encoding completed.")
  ev_encoded_df.to_csv(os.path.join(gold_dir, 'electric_vehicles_one_hot_encoded.csv'), index=False)


if __name__ == "__main__":
    main()