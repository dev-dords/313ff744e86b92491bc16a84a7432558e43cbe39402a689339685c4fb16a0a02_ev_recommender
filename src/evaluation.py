import joblib
import pandas as pd
import os

src_dir = os.getcwd()
data_dir = os.path.join(src_dir, 'data')
gold_dir = os.path.join(data_dir, 'gold')


def main():
  model_path = os.path.join(src_dir, 'models', 'model.pkl')
  knn = joblib.load(model_path)

  data = pd.read_csv(os.path.join(gold_dir, 'electric_vehicles_test_processed.csv'))

  X_test = data.drop(columns=['model'], errors='ignore')
  y_test = data['model']

  distances, indices = knn.kneighbors(X_test, n_neighbors=5)

  report_path = os.path.join(src_dir, 'reports', 'metrics.txt')
  os.makedirs(os.path.dirname(report_path), exist_ok=True)
  with open(report_path, 'w') as f:
      for i in range(5):
          f.write(f"\nQuery Vehicle: {y_test.iloc[i]}\n")
          f.write("Top 5 Similar Vehicles:\n")
          for rank, idx in enumerate(indices[i]):
              sim_model = y_test.iloc[idx]
              sim_dist = distances[i][rank]
              f.write(f"  {rank + 1}. {sim_model} (distance: {sim_dist:.4f})\n")

if __name__ == "__main__":
    main()
    print("\nRecommendation generation completed.")





