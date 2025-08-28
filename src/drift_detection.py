import glob
import json
from typing import Any, Dict

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

import mlflow

BASE_DIR = "/app"
DATA_DIR = f"{BASE_DIR}/data"
BRONZE_DIR = f"{DATA_DIR}/bronze"
SILVER_DIR = f"{DATA_DIR}/silver"
GOLD_DIR = f"{DATA_DIR}/gold"
DATA_FILE = f"{BRONZE_DIR}/electric_vehicles_spec_2025.csv.csv"


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pandas.DataFrame: Loaded data as a DataFrame.
    """
    csv_files = glob.glob(
        f"{file_path}/*.csv") if not file_path.endswith(".csv") else [file_path]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {file_path}")
    df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
    electric_vehicles = pd.concat(df_list, ignore_index=True)
    return electric_vehicles


def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    reference_data = load_data(reference_data_path)
    reference_data = reference_data[reference_data.columns.difference([
                                                                      'model'])]
    current_data = load_data(current_data_path)
    current_data = current_data[current_data.columns.difference(['model'])]

    report = Report([DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    report.save_json("/app/reports/drift_report.json")
    # Use the get_dict() method to retrieve the report as a dictionary
    # report_dict = report.get_dict()

    # dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
    # feature_drift = report_dict["metrics"][1]["result"]["drift_by_columns"]

    # feature_scores = [
    #     col_info.get("drift_score", 0.0)
    #     for col_info in feature_drift.values()
    #     if "drift_score" in col_info
    # ]
    # overall_drift_score = sum(feature_scores) / len(feature_scores) if feature_scores else 0.0

    # return {
    #     "drift_detected": dataset_drift,
    #     "feature_drift": feature_drift,
    #     "overall_drift_score": overall_drift_score
    # }


def init_detect_drift():
    mlflow.set_tracking_uri("http://mlflow:5000")
    print("Analyzing data drift...")
    reference_data_path = f'{SILVER_DIR}/silver_non_drift/'
    current_data_path = f'{SILVER_DIR}/'

    # Call the corrected function and store the result
    detect_drift(reference_data_path, current_data_path)

    # # Log drift status to MLFlow
    # mlflow.log_param("test_drift_detected", test_drift_results["drift_detected"])
    # mlflow.log_param("test_overall_drift_score", test_drift_results["overall_drift_score"])

    # # Raise error if drift detected
    # if test_drift_results["drift_detected"]:
    #   raise ValueError("Data drift detected in test data")

    print("Data drift analysis completed successfully!")
