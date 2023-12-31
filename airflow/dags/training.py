import pendulum
import os
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG(
    'car_training_pipeline',
    default_args={'retries': 2},
    description='Car-Price-Prediction',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2023, 7, 5, tz="UTC"),
    catchup=False,
    tags=['car_training_pipeline']
) as dag:


    def training(**kwargs):
        from CarPrice.pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()

    
    def sync_artifact_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        artifact_folder = "/app/artifact"
        saved_model = "/app/saved_models"
        os.system(f"aws s3 sync {artifact_folder} s3://{bucket_name}/artifacts")
        os.system(f"aws s3 sync {saved_model} s3://{bucket_name}/saved_models")

    # Commence training
    training_pipeline = PythonOperator(
        task_id="car_training_pipeline",
        python_callable=training
    )

    # Sync the relevant objects to the S3 bucket
    sync_data_to_s3 = PythonOperator(
        task_id="sync_data_to_s3",
        python_callable=sync_artifact_to_s3_bucket
    )

    # flow
    training_pipeline >> sync_data_to_s3