import os
import pandas as pd
import numpy as np


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

def fix_missing_values(df):
    """
    Fix missing values in the DataFrame.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to fix.
    
    Returns:
    pandas.DataFrame: DataFrame with missing values fixed.
    """
    electric_vehicles = df.copy()
    electric_vehicles = electric_vehicles.dropna(subset=['model'])
    electric_vehicles = electric_vehicles.drop(columns=['number_of_cells'])
    electric_vehicles.loc[electric_vehicles['fast_charging_power_kw_dc'].isna(), 'fast_charging_power_kw_dc'] = 80
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
    electric_vehicles = fix_missing_values(electric_vehicles)
    save_data(electric_vehicles, os.path.join(silver_dir, 'electric_vehicles_preprocessed.csv'))
    
if __name__ == "__main__":
    main()
