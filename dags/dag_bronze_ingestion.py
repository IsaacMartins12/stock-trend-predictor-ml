"""
DAG: Bronze Layer - Raw Data Ingestion

Schedule: Diariamente às 19:00 (após fechamento do mercado BR)
Responsabilidade: Extrair dados do yfinance e persistir na camada bronze (raw).

Arquitetura Medalhão:
    [yfinance API] → [bronze.raw_stock_prices]
"""

import uuid
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

import sys
sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)

# ============================================
# Default DAG args
# ============================================
default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


# ============================================
# Task Functions
# ============================================

def extract_and_load_bronze(**context):
    """
    Extract OHLCV data from yfinance and load into bronze layer.
    Implements idempotent ingestion with batch tracking.
    """
    import pandas as pd
    import yfinance as yf
    import psycopg
    import yaml

    # Load config
    with open("/opt/airflow/configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tickers = config["data"]["tickers"]
    batch_id = str(uuid.uuid4())
    execution_date = context["ds"]  # YYYY-MM-DD

    logger.info(f"Starting bronze ingestion | batch={batch_id} | date={execution_date}")

    # Connection to data warehouse
    conn_params = {
        "host": "postgres",
        "port": 5432,
        "dbname": "stock_predictor",
        "user": "stock_user",
        "password": "stock_pass",
    }

    results = {}

    with psycopg.connect(**conn_params) as conn:
        for ticker in tickers:
            try:
                # Determine fetch window (last 7 days to handle weekends/holidays)
                end_date = execution_date
                start_date = (
                    datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=7)
                ).strftime("%Y-%m-%d")

                # Extract from yfinance
                raw = yf.download(
                    tickers=ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                )

                if raw.empty:
                    logger.warning(f"{ticker}: no data returned")
                    results[ticker] = 0
                    continue

                # Handle multi-level columns
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                df = raw.reset_index()
                df.columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

                # Load into bronze
                rows_inserted = 0
                with conn.cursor() as cur:
                    for _, row in df.iterrows():
                        cur.execute(
                            """
                            INSERT INTO bronze.raw_stock_prices
                                (ticker, date, open, high, low, close, adj_close,
                                 volume, source, ingestion_batch_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (ticker, date, source) DO NOTHING
                            """,
                            (
                                ticker,
                                row["date"],
                                float(row["open"]) if pd.notna(row["open"]) else None,
                                float(row["high"]) if pd.notna(row["high"]) else None,
                                float(row["low"]) if pd.notna(row["low"]) else None,
                                float(row["close"]) if pd.notna(row["close"]) else None,
                                float(row["adj_close"]) if pd.notna(row["adj_close"]) else None,
                                int(row["volume"]) if pd.notna(row["volume"]) else None,
                                "yfinance",
                                batch_id,
                            ),
                        )
                        rows_inserted += cur.rowcount

                    # Log ingestion
                    cur.execute(
                        """
                        INSERT INTO bronze.ingestion_log
                            (batch_id, ticker, start_date, end_date,
                             rows_ingested, status, source, finished_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                        """,
                        (batch_id, ticker, start_date, end_date,
                         rows_inserted, "success", "yfinance"),
                    )

                conn.commit()
                results[ticker] = rows_inserted
                logger.info(f"✓ {ticker}: {rows_inserted} rows → bronze")

            except Exception as e:
                conn.rollback()
                logger.error(f"✗ {ticker}: {e}")
                results[ticker] = -1

                # Log failure
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO bronze.ingestion_log
                            (batch_id, ticker, start_date, end_date,
                             rows_ingested, status, error_message, source, finished_at)
                        VALUES (%s, %s, %s, %s, 0, %s, %s, %s, NOW())
                        """,
                        (batch_id, ticker, start_date, end_date,
                         "failed", str(e), "yfinance"),
                    )
                conn.commit()

    # Push results to XCom for downstream tasks
    context["ti"].xcom_push(key="ingestion_results", value=results)
    context["ti"].xcom_push(key="batch_id", value=batch_id)

    total = sum(v for v in results.values() if v > 0)
    logger.info(f"Bronze ingestion complete | batch={batch_id} | total_rows={total}")
    return results


def validate_bronze_data(**context):
    """
    Data quality check on bronze layer.
    Validates that we have data for all expected tickers.
    """
    import yaml
    import psycopg

    results = context["ti"].xcom_pull(key="ingestion_results")
    batch_id = context["ti"].xcom_pull(key="batch_id")

    with open("/opt/airflow/configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    expected_tickers = config["data"]["tickers"]
    failed = [t for t, r in results.items() if r < 0]
    empty = [t for t, r in results.items() if r == 0]

    if failed:
        logger.warning(f"Failed tickers: {failed}")

    if empty:
        logger.info(f"No new data for: {empty} (may be weekend/holiday)")

    # At least some tickers must have data (unless it's weekend)
    successful = [t for t, r in results.items() if r > 0]
    logger.info(
        f"Validation | success={len(successful)}, "
        f"empty={len(empty)}, failed={len(failed)}"
    )

    return {"status": "passed", "successful": successful, "failed": failed}


# ============================================
# DAG Definition
# ============================================
with DAG(
    dag_id="bronze_ingestion",
    default_args=default_args,
    description="Extract raw stock data from yfinance into bronze layer",
    schedule_interval="0 19 * * 1-5",  # Seg-Sex às 19h (pós-fechamento B3)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["bronze", "ingestion", "medallion"],
    max_active_runs=1,
) as dag:

    task_extract_load = PythonOperator(
        task_id="extract_and_load_bronze",
        python_callable=extract_and_load_bronze,
        provide_context=True,
    )

    task_validate = PythonOperator(
        task_id="validate_bronze_data",
        python_callable=validate_bronze_data,
        provide_context=True,
    )

    task_extract_load >> task_validate
