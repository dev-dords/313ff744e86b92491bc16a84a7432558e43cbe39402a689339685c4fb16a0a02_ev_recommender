import os
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

src_dir = os.getcwd()
data_dir = os.path.join(src_dir, 'data')
bronze_dir = os.path.join(data_dir, 'bronze')
silver_dir = os.path.join(data_dir, 'silver')
data_file = os.path.join(bronze_dir, 'electric_vehicles_spec_2025.csv.csv')

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pandas.DataFrame: Loaded data as a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    electric_vehicles = pd.read_csv(file_path)
    print(f"Loading data from {file_path}...")
    print(f"Data has shape: {electric_vehicles.shape}")
    return electric_vehicles

def check_missing_values(df):
    """
    Check for missing values in the DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame to check.

    Returns:
    pandas.Series: Series with counts of missing values per column.
    """
    missing_values = df.isnull().sum()
    print("Missing values in each column:")
    print(missing_values[missing_values > 0])

def fill_na_discrepancy(df):
    """
    Fix missing values in the DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame to fix.

    Returns:
    pandas.DataFrame: DataFrame with missing values fixed.
    """

    def replace_banana(match):
        number = int(match.group(1))
        return str(number * 50)

    electric_vehicles = df.copy()
    electric_vehicles = electric_vehicles.dropna(subset=['model'])
    electric_vehicles = electric_vehicles.drop(columns=['number_of_cells', 'source_url', 'battery_type'])
    electric_vehicles.loc[electric_vehicles['fast_charging_power_kw_dc'].isna(), 'fast_charging_power_kw_dc'] = 80
    electric_vehicles.loc[electric_vehicles['torque_nm'].isna(), 'torque_nm'] = 0
    electric_vehicles.loc[electric_vehicles['fast_charge_port'].isna(), 'fast_charge_port'] = 'CCS'
    brand_means = electric_vehicles.groupby('brand').agg({
        'towing_capacity_kg': 'mean'
    })
    brand_means.fillna(0, inplace=True)
    brand_means_dict = brand_means['towing_capacity_kg'].to_dict()
    electric_vehicles['towing_capacity_kg'] = electric_vehicles['towing_capacity_kg'].fillna(
        electric_vehicles['brand'].map(brand_means_dict)
    )
    electric_vehicles.loc[ electric_vehicles['brand']=='Ford', 'cargo_volume_l'] = 571
    mask = electric_vehicles['cargo_volume_l'].str.contains(r'(?i)Banana Boxes', na=False)
    electric_vehicles.loc[mask, 'cargo_volume_l'] = electric_vehicles.loc[mask, 'cargo_volume_l'].str.replace(
        r'(?i)(\d+)\s*Banana Boxes', replace_banana, regex=True
    )

    return electric_vehicles

def save_data(df, file_path):
    """
    Save DataFrame to a CSV file.

    Parameters:
    df (pandas.DataFrame): DataFrame to save.
    file_path (str): Path to save the CSV file.
    """
    df.to_csv(file_path)
    print(f"Data saved to {file_path}")

def main():
    print("Data preprocessing module loaded successfully!")
    electric_vehicles = load_data(data_file)
    check_missing_values(electric_vehicles)
    electric_vehicles = fill_na_discrepancy(electric_vehicles)

    X = electric_vehicles.drop('model', axis=1)
    y = electric_vehicles['model']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    training_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    save_data(training_data, os.path.join(silver_dir, 'electric_vehicles_training_data.csv'))
    save_data(test_data, os.path.join(silver_dir, 'electric_vehicles_test_data.csv'))
if __name__ == "__main__":
    main()
