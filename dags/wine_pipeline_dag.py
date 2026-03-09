from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'wine_quality_training_pipeline',
    default_args=default_args,
    description='Pipeline for training wine quality model',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # Чтение пути к скрипту из переменных Airflow (со значением по умолчанию)
    train_script = Variable.get("train_script_path", default_var="train.py")

    load_data_task = BashOperator(
        task_id='load_data',
        bash_command='dvc pull data/wine_quality.csv.dvc || echo "Data is already loaded or DVC not reachable"',
    )

    train_model_task = BashOperator(
        task_id='train_model',
        bash_command=f'python {train_script}',
    )

    save_model_task = BashOperator(
        task_id='save_model_to_dvc',
        bash_command='dvc add models/wine_quality_model.pkl && git add models/wine_quality_model.pkl.dvc && git commit -m "Update model via Airflow" || echo "No changes to commit"',
    )

    load_data_task >> train_model_task >> save_model_task
