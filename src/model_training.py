import os

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.neighbors import NearestNeighbors

import mlflow
from data_preprocessing import load_data

src_dir = os.getcwd()
model_dir = os.path.join(src_dir, 'models')
data_dir = os.path.join(src_dir, 'data')
gold_dir = os.path.join(data_dir, 'gold')
artifact_path = os.path.join(src_dir, 'mlflow', 'artifacts')


class KNNModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc model wrapper for your trained model.
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
            with open(context.artifacts["feature_names"], 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        if self.preprocessor:
            processed_input = self.preprocessor.transform(model_input)
        else:
            processed_input = model_input.values
        return self.model.kneighbors(processed_input)


def train_model():
    """
    Placeholder function for model training.
    This function should be implemented to train a machine learning model.
    """

    ev_train = load_data(os.path.join(
        gold_dir, 'electric_vehicles_train_processed.csv'))
    X_train = ev_train.drop(columns=['model'])
    y_train = ev_train['model']

    n_neighbors = 5
    metric = 'cosine'
    algorithm = 'brute'

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("EV_Model_Training")

    with mlflow.start_run():
        knn = NearestNeighbors(n_neighbors=n_neighbors,
                               metric=metric, algorithm=algorithm)
        knn.fit(X_train)

        os.makedirs(artifact_path, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'model.pkl')
        joblib.dump(knn, model_path)

        artifact_model_path = os.path.join(artifact_path, 'model.pkl')
        joblib.dump(knn, artifact_model_path)

        feature_names_path = os.path.join(artifact_path, 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            f.writelines(f"{name}\n" for name in X_train.columns)

        distances, _ = knn.kneighbors(X_train.head(5))
        signature = infer_signature(X_train, distances)

        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("metric", metric)
        mlflow.log_param("algorithm", algorithm)

        mlflow.pyfunc.log_model(
            artifact_path="knn_model",
            python_model=KNNModelWrapper(),
            artifacts={
                "model": artifact_model_path,
                "feature_names": feature_names_path
            },
            signature=signature,
        )


def main():
    print("Starting model training...")
    train_model()
    print("Model training completed and saved to models/model.pkl")


if __name__ == "__main__":
    main()
