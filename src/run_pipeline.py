from data_preprocessing import preprocess_data
from evaluation import evaluate
from feature_engineering import feature_engineer
from model_training import train_model


def main():
    """
    Main function to run the data preprocessing and feature engineering pipelines.
    """
    print("Starting the end-to-end pipeline...")
    print("Step 1: Data Preprocessing")
    preprocess_data()
    print("Step 2: Feature Engineering")
    feature_engineer()
    print("Step 3: Model Training")
    train_model()
    print("Step 4: Evaluation")
    evaluate()


if __name__ == "__main__":
    main()
