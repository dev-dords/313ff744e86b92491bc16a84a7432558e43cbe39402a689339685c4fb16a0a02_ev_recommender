import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from data_preprocessing import load_data, save_data

# Project root inside container
BASE_DIR = "/app"
DATA_DIR = f"{BASE_DIR}/data"
SILVER_DIR = f"{DATA_DIR}/silver"
GOLD_DIR = f"{DATA_DIR}/gold"


def feature_engineering_pipeline(training_data, test_data):
    columns_to_scale = [
        "top_speed_kmh", "battery_capacity_kWh", "torque_nm",
        "efficiency_wh_per_km", "range_km", "acceleration_0_100_s",
        "fast_charging_power_kw_dc", "towing_capacity_kg",
        "cargo_volume_l", "seats", "length_mm", "width_mm", "height_mm"
    ]
    categorical_columns = [
        "brand", "fast_charge_port", "segment", "drivetrain", "car_body_type"
    ]

    X_train = training_data.drop("model", axis=1)
    y_train = training_data["model"]
    X_test = test_data.drop("model", axis=1)
    y_test = test_data["model"]

    # Define preprocessors
    numeric_transformer = MinMaxScaler()
    categorical_transformer = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, columns_to_scale),
            ("cat", categorical_transformer, categorical_columns),
        ],
        remainder="passthrough",
    )

    train_processed = preprocessor.fit_transform(X_train)
    test_processed = preprocessor.transform(X_test)

    # Get feature names after transformation
    cat_feature_names = preprocessor.named_transformers_[
        "cat"].get_feature_names_out(categorical_columns)
    all_feature_names = (
        columns_to_scale
        + list(cat_feature_names)
        + [col for col in X_train.columns if col not in columns_to_scale +
            categorical_columns]
    )

    return (
        pd.DataFrame(train_processed, columns=all_feature_names),
        pd.DataFrame(test_processed, columns=all_feature_names),
    )


def feature_engineer():
    training_data = load_data(
        f"{SILVER_DIR}/electric_vehicles_training_data.csv")
    test_data = load_data(f"{SILVER_DIR}/electric_vehicles_test_data.csv")

    train_processed, test_processed = feature_engineering_pipeline(
        training_data, test_data)

    train_processed = pd.concat(
        [train_processed, training_data["model"]], axis=1)
    test_processed = pd.concat([test_processed, test_data["model"]], axis=1)

    # Drop index columns if present
    for df in (train_processed, test_processed):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    save_data(train_processed,
              f"{GOLD_DIR}/electric_vehicles_train_processed.csv")
    save_data(test_processed,
              f"{GOLD_DIR}/electric_vehicles_test_processed.csv")


if __name__ == "__main__":
    print("Feature engineering module loaded successfully!")
    feature_engineer()
    print("Feature engineering pipeline completed successfully!")
