from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from src.data_preprocessing import preprocess_data
from src.evaluation import evaluate
from src.feature_engineering import feature_engineer
from src.model_training import train_model


def create_dag():
    with DAG(
        dag_id="ml_pipeline",
        start_date=datetime(2022, 1, 1),
        schedule=None,
        catchup=False,
        tags=["ml"],
    ) as dag:
        print('Creating ML Pipeline DAG')
        preprocess_task = PythonOperator(
            task_id="preprocess_data",
            python_callable=preprocess_data,
        )

        feature_engineering_task = PythonOperator(
            task_id="feature_engineer",
            python_callable=feature_engineer,
        )

        train_model_task = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
        )

        evaluate_task = PythonOperator(
            task_id="evaluate_model",
            python_callable=evaluate,
        )

        preprocess_task >> feature_engineering_task >> train_model_task >> evaluate_task

        return dag


dag = create_dag()
