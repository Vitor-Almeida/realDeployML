from airflow.decorators import dag,task
from airflow.utils.dates import days_ago
import realdeployml._train as train

# Default DAG args
default_args = {
    "owner": "airflow",
}

@dag(
    dag_id="canaryFlow",
    description="MLOPS testing of a canary deployment using airflow",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["example"],
)

def example():

    @task
    def task_1():
        train.main()
        return None

    # Task relationships
    task_1()
    #data = task_1()
    #model = task_2(data=data)
    #task_3(model=model,data=data)

example_dag = example()