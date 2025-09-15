from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.datasets import Dataset
from datetime import datetime

ORDERS_ENRICHED_DS = Dataset("delta://silver/order_payments_enriched")

with DAG(
    dag_id="ml_export_snapshots",
    start_date=datetime(2025, 8, 1),
    schedule=[ORDERS_ENRICHED_DS],
    catchup=False,
    max_active_runs=1,
    tags=["ecommerce", "ml", "snapshots"],
) as dag:

    # Write dated Parquet/Delta under your repo-mounted path
    export_cmd = r"""
    /opt/spark/bin/spark-sql \
      --master spark://spark-master:7077 \
      -e "
      -- Interactions (purchases)
      CREATE TABLE IF NOT EXISTS delta.ecom.ml_interactions_{{ ds_nodash }}
      WITH (location = 'file:///workspace/data/delta/ecom/ml/interactions/{{ ds }}') AS
      SELECT
        o.user_id,
        CAST(i.product_id AS VARCHAR) AS item_id,
        o.event_ts            AS event_time,
        'purchase'            AS event_type
      FROM silver.orders_clean o
      CROSS JOIN UNNEST(o.items) AS t(i);

      -- If you have a dim_items table later, add an items snapshot here.
      "
    """
    BashOperator(task_id="export_for_ml", bash_command=export_cmd)
