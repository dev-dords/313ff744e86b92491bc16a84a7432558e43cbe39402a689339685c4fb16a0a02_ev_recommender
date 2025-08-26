import os

import joblib
import pandas as pd
from sklearn.metrics import silhouette_score

src_dir = os.getcwd()
data_dir = os.path.join(src_dir, 'data')
gold_dir = os.path.join(data_dir, 'gold')


def evaluate():
    model_path = os.path.join(src_dir, 'models', 'model.pkl')
    knn = joblib.load(model_path)

    data = pd.read_csv(os.path.join(
        gold_dir, 'electric_vehicles_test_processed.csv'))

    X_test = data.drop(columns=['model'], errors='ignore')
    labels = knn.predict(X_test)

    # Compute metrics
    inertia = knn.inertia_ if hasattr(knn, 'inertia_') else None
    silhouette = silhouette_score(X_test, labels)

    report_path = os.path.join(src_dir, 'reports', 'metrics.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("Clustering Evaluation Metrics\n")
        f.write("============================\n")
        if inertia is not None:
            f.write(f"Inertia: {inertia:.4f}\n")
        else:
            f.write("Inertia: N/A (model has no inertia_ attribute)\n")
        f.write(f"Silhouette Score: {silhouette:.4f}\n")

    return {"silhouette_score": silhouette}, "clustering"


if __name__ == "__main__":
    print("Starting evaluation...")
    metrics, task_type = evaluate()
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
