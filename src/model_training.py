import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.cluster import KMeans

import mlflow
from data_preprocessing import load_data

# Fixed paths inside container
BASE_DIR = "/app"
MODEL_DIR = f"{BASE_DIR}/models"
DATA_DIR = f"{BASE_DIR}/data"
GOLD_DIR = f"{DATA_DIR}/gold"
ARTIFACT_DIR = f"{BASE_DIR}/mlflow/artifacts"


class KMeansModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc model wrapper for trained KMeans.
    Encapsulates preprocessing and prediction logic.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
        if "preprocessor" in context.artifacts:
            self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        if "feature_names" in context.artifacts:
            with open(context.artifacts["feature_names"], "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        if self.preprocessor:
            processed_input = self.preprocessor.transform(model_input)
        else:
            processed_input = model_input.values
        return self.model.predict(processed_input)


def train_model():
    """
    Train a KMeans clustering model for EV data and log it to MLflow.
    """
    # --- Load data ---
    ev_train = load_data(f"{GOLD_DIR}/electric_vehicles_train_processed.csv")
    X_train = ev_train.drop(columns=["model"], errors="ignore")

    # --- Model setup ---
    n_clusters = 6
    random_state = 42

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("EV_Model_Training")

    with mlflow.start_run():
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(X_train)

        model_path = f"{MODEL_DIR}/model.pkl"
        joblib.dump(kmeans, model_path)

        artifact_model_path = f"{ARTIFACT_DIR}/model.pkl"
        joblib.dump(kmeans, artifact_model_path)

        feature_names_path = f"{ARTIFACT_DIR}/feature_names.txt"
        with open(feature_names_path, "w") as f:
            f.writelines(f"{name}\n" for name in X_train.columns)

        # --- Metrics & signature ---
        labels = kmeans.predict(X_train.head(5))
        signature = infer_signature(X_train, labels)

        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("random_state", random_state)

        # --- Log model to MLflow ---
        mlflow.pyfunc.log_model(
            artifact_path="kmeans_model",
            python_model=KMeansModelWrapper(),
            artifacts={
                "model": artifact_model_path,
                "feature_names": feature_names_path,
            },
            signature=signature,
        )


def main():
    print("Starting model training...")
    train_model()
    print("Model training completed and saved to /app/models/model.pkl")


if __name__ == "__main__":
    main()
