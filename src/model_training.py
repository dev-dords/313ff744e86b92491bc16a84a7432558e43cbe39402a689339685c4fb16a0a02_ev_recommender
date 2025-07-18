import os

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from data_preprocessing import load_data

src_dir = os.getcwd()
model_dir = os.path.join(src_dir, 'models')
data_dir = os.path.join(src_dir, 'data')
gold_dir = os.path.join(data_dir, 'gold')


def train_model():
    """
    Placeholder function for model training.
    This function should be implemented to train a machine learning model.
    """

    ev_train = load_data(os.path.join(
        gold_dir, 'electric_vehicles_train_processed.csv'))
    X_train = ev_train.drop(columns=['model'])
    y_train = ev_train['model']

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(X_train)
    joblib.dump(knn, os.path.join(model_dir, 'model.pkl'))


def main():
    print("Starting model training...")
    train_model()
    print("Model training completed and saved to models/model.pkl")


if __name__ == "__main__":
    main()
