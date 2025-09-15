from airflow import DAG
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from datetime import datetime

with DAG("ml_export_snapshots", start_date=datetime(2025,9,1), schedule_interval="@daily", catchup=False) as dag:
    date = "{{ ds }}"

    export_interactions = SQLExecuteQueryOperator(
        task_id="export_interactions",
        conn_id="trino_default",
        sql=f"""
        CREATE TABLE IF NOT EXISTS delta.ecom.ml_interactions_{ { '{' }{ 'ds_nodash' }{ '}' } }
        WITH (location = 'file:///workspace/data/delta/ecom/ml/interactions/{ { '{' }{ 'ds' }{ '}' } }') AS
        SELECT user_id, item_id, event_time, event_type
        FROM delta.ecom.silver_orders
        WHERE event_time < TIMESTAMP '{ { '{' }{ 'ds' }{ '}' } } 00:00:00';
        """,
    )

    export_items = SQLExecuteQueryOperator(
        task_id="export_items",
        conn_id="trino_default",
        sql=f"""
        CREATE TABLE IF NOT EXISTS delta.ecom.ml_items_{ { '{' }{ 'ds_nodash' }{ '}' } }
        WITH (location = 'file:///workspace/data/delta/ecom/ml/items/{ { '{' }{ 'ds' }{ '}' } }') AS
        SELECT item_id, category, brand, price, title, description
        FROM delta.ecom.dim_items;
        """,
    )

    export_interactions >> export_items
