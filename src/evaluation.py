import joblib
import pandas as pd
from sklearn.metrics import silhouette_score


def evaluate():
    # --- Load model ---
    model_path = "/app/models/model.pkl"
    kmeans = joblib.load(model_path)

    # --- Load test data ---
    test_file = "/app/data/gold/electric_vehicles_test_processed.csv"
    data = pd.read_csv(test_file)

    # drop label column if it exists
    X_test = data.drop(columns=["model"], errors="ignore")

    # --- Predict cluster labels ---
    labels = kmeans.predict(X_test)

    # --- Compute metrics ---
    inertia = getattr(kmeans, "inertia_", None)
    silhouette = silhouette_score(X_test, labels)

    # --- Write report ---
    report_path = "/app/reports/metrics.txt"
    with open(report_path, "w") as f:
        f.write("Clustering Evaluation Metrics\n")
        f.write("============================\n")
        f.write(
            f"Inertia: {inertia:.4f}\n" if inertia is not None else "Inertia: N/A\n")
        f.write(f"Silhouette Score: {silhouette:.4f}\n")

    return {"silhouette_score": silhouette}, "clustering"


if __name__ == "__main__":
    print("Starting evaluation...")
    metrics, task_type = evaluate()
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
