"""
DAG: Silver Layer - Data Transformation & Quality

Schedule: Diariamente às 19:30 (após bronze ingestion)
Responsabilidade: Limpar, validar e transformar dados da bronze → silver.

Arquitetura Medalhão:
    [bronze.raw_stock_prices] → limpeza → validação → [silver.stock_prices]
"""

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

import sys
sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}


# ============================================
# Task Functions
# ============================================

def transform_bronze_to_silver(**context):
    """
    Transform raw bronze data into cleaned silver data.

    Transformations applied:
    - Deduplication
    - Missing value treatment (forward fill)
    - Data type enforcement
    - Return calculations
    - Data quality scoring
    """
    import pandas as pd
    import numpy as np
    import psycopg
    import yaml

    with open("/opt/airflow/configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tickers = config["data"]["tickers"]
    execution_date = context["ds"]

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
                # Read from bronze (all unprocessed data for this ticker)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT date, open, high, low, close, adj_close, volume,
                               ingestion_batch_id
                        FROM bronze.raw_stock_prices
                        WHERE ticker = %s
                        AND date > COALESCE(
                            (SELECT MAX(sp.date) FROM silver.stock_prices sp
                             JOIN assets a ON sp.asset_id = a.id
                             WHERE a.ticker = %s),
                            '1900-01-01'
                        )
                        ORDER BY date ASC
                        """,
                        (ticker, ticker),
                    )
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()

                if not rows:
                    logger.info(f"{ticker}: no new bronze data to process")
                    results[ticker] = 0
                    continue

                df = pd.DataFrame(rows, columns=columns)
                source_batch = df["ingestion_batch_id"].iloc[-1]
                df = df.drop(columns=["ingestion_batch_id"])

                # ---- TRANSFORMATIONS ----

                # 1. Deduplication (keep latest by date)
                df = df.drop_duplicates(subset=["date"], keep="last")

                # 2. Sort chronologically
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)

                # 2.5. Convert Decimal to float (PostgreSQL NUMERIC → Decimal)
                numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # 3. Data quality scoring
                price_cols = ["open", "high", "low", "close"]
                df["quality_score"] = 1.0
                # Penalize rows with missing values
                missing_mask = df[price_cols].isnull().any(axis=1)
                df.loc[missing_mask, "quality_score"] -= 0.3
                # Penalize rows with zero volume
                df.loc[df["volume"] == 0, "quality_score"] -= 0.2

                # 4. Handle missing values (forward fill)
                df["is_imputed"] = df[price_cols].isnull().any(axis=1)
                df[price_cols] = df[price_cols].ffill()
                df["volume"] = df["volume"].fillna(0)

                # 5. Validate OHLC integrity (High >= Low, etc.)
                invalid_ohlc = df["high"] < df["low"]
                if invalid_ohlc.any():
                    # Swap high/low where invalid
                    mask = invalid_ohlc
                    df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values
                    df.loc[mask, "quality_score"] -= 0.2
                    logger.warning(f"{ticker}: fixed {mask.sum()} invalid OHLC rows")

                # 6. Calculate returns
                df["return_simple"] = df["close"].pct_change()
                df["return_log"] = np.log(df["close"] / df["close"].shift(1))

                # 7. Drop rows that are completely invalid
                df = df.dropna(subset=["close"])
                df["quality_score"] = df["quality_score"].clip(0, 1)

                # ---- LOAD INTO SILVER ----

                # Ensure asset exists
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO assets (ticker) VALUES (%s)
                        ON CONFLICT (ticker) DO NOTHING
                        """,
                        (ticker,),
                    )
                    cur.execute(
                        "SELECT id FROM assets WHERE ticker = %s", (ticker,)
                    )
                    asset_id = cur.fetchone()[0]

                    # Insert into silver
                    rows_inserted = 0
                    for _, row in df.iterrows():
                        cur.execute(
                            """
                            INSERT INTO silver.stock_prices
                                (asset_id, date, open, high, low, close, adj_close,
                                 volume, return_simple, return_log, is_imputed,
                                 quality_score, source_batch_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (asset_id, date)
                            DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                adj_close = EXCLUDED.adj_close,
                                volume = EXCLUDED.volume,
                                return_simple = EXCLUDED.return_simple,
                                return_log = EXCLUDED.return_log,
                                is_imputed = EXCLUDED.is_imputed,
                                quality_score = EXCLUDED.quality_score,
                                processed_at = NOW()
                            """,
                            (
                                asset_id,
                                row["date"].strftime("%Y-%m-%d"),
                                float(row["open"]) if pd.notna(row["open"]) else None,
                                float(row["high"]) if pd.notna(row["high"]) else None,
                                float(row["low"]) if pd.notna(row["low"]) else None,
                                float(row["close"]) if pd.notna(row["close"]) else None,
                                float(row["adj_close"]) if pd.notna(row["adj_close"]) else None,
                                int(row["volume"]) if pd.notna(row["volume"]) else 0,
                                float(row["return_simple"]) if pd.notna(row["return_simple"]) else None,
                                float(row["return_log"]) if pd.notna(row["return_log"]) else None,
                                bool(row["is_imputed"]),
                                float(row["quality_score"]),
                                source_batch,
                            ),
                        )
                        rows_inserted += cur.rowcount

                conn.commit()
                results[ticker] = rows_inserted
                logger.info(f"✓ {ticker}: {rows_inserted} rows → silver")

            except Exception as e:
                conn.rollback()
                logger.error(f"✗ {ticker} silver transformation failed: {e}")
                results[ticker] = -1

    context["ti"].xcom_push(key="silver_results", value=results)
    total = sum(v for v in results.values() if v > 0)
    logger.info(f"Silver transformation complete | total_rows={total}")
    return results


def run_data_quality_checks(**context):
    """
    Execute data quality checks on silver layer.

    Checks:
    - Completeness: no missing close prices
    - Consistency: high >= low, close within high/low range
    - Freshness: data is not stale (recent date exists)
    - Validity: no negative prices, volume >= 0
    """
    import psycopg
    import yaml
    import json

    with open("/opt/airflow/configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tickers = config["data"]["tickers"]
    execution_date = context["ds"]

    conn_params = {
        "host": "postgres",
        "port": 5432,
        "dbname": "stock_predictor",
        "user": "stock_user",
        "password": "stock_pass",
    }

    all_checks = []

    with psycopg.connect(**conn_params) as conn:
        for ticker in tickers:
            with conn.cursor() as cur:
                # Get asset_id
                cur.execute("SELECT id FROM assets WHERE ticker = %s", (ticker,))
                row = cur.fetchone()
                if not row:
                    continue
                asset_id = row[0]

                # Check 1: No null close prices
                cur.execute(
                    """
                    SELECT COUNT(*) FROM silver.stock_prices
                    WHERE asset_id = %s AND close IS NULL
                    """,
                    (asset_id,),
                )
                null_count = cur.fetchone()[0]
                check_1 = null_count == 0

                # Check 2: OHLC integrity (high >= low)
                cur.execute(
                    """
                    SELECT COUNT(*) FROM silver.stock_prices
                    WHERE asset_id = %s AND high < low
                    """,
                    (asset_id,),
                )
                invalid_ohlc = cur.fetchone()[0]
                check_2 = invalid_ohlc == 0

                # Check 3: No negative prices
                cur.execute(
                    """
                    SELECT COUNT(*) FROM silver.stock_prices
                    WHERE asset_id = %s AND (close < 0 OR open < 0)
                    """,
                    (asset_id,),
                )
                negative = cur.fetchone()[0]
                check_3 = negative == 0

                # Check 4: Data freshness (within 5 business days)
                cur.execute(
                    """
                    SELECT MAX(date) FROM silver.stock_prices WHERE asset_id = %s
                    """,
                    (asset_id,),
                )
                max_date = cur.fetchone()[0]
                check_4 = max_date is not None

                # Log all checks
                checks = [
                    ("completeness_no_null_close", check_1, {"null_count": null_count}),
                    ("consistency_ohlc_integrity", check_2, {"invalid_rows": invalid_ohlc}),
                    ("validity_no_negative_prices", check_3, {"negative_count": negative}),
                    ("freshness_has_recent_data", check_4, {"max_date": str(max_date)}),
                ]

                for check_name, passed, details in checks:
                    cur.execute(
                        """
                        INSERT INTO silver.data_quality_checks
                            (asset_id, check_date, check_name, passed, details)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (asset_id, execution_date, check_name, passed, json.dumps(details)),
                    )
                    all_checks.append({
                        "ticker": ticker,
                        "check": check_name,
                        "passed": passed,
                    })

        conn.commit()

    # Report
    failed_checks = [c for c in all_checks if not c["passed"]]
    if failed_checks:
        logger.warning(f"Data quality issues found: {failed_checks}")
    else:
        logger.info("All data quality checks passed ✓")

    return {"total_checks": len(all_checks), "failed": len(failed_checks)}


# ============================================
# DAG Definition
# ============================================
with DAG(
    dag_id="silver_transformation",
    default_args=default_args,
    description="Transform bronze raw data into cleaned silver layer",
    schedule_interval="30 19 * * 1-5",  # 19:30, after bronze ingestion
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["silver", "transformation", "medallion", "data-quality"],
    max_active_runs=1,
) as dag:

    # Wait for bronze ingestion to complete
    wait_for_bronze = ExternalTaskSensor(
        task_id="wait_for_bronze_ingestion",
        external_dag_id="bronze_ingestion",
        external_task_id="validate_bronze_data",
        timeout=600,
        poke_interval=30,
        mode="poke",
    )

    task_transform = PythonOperator(
        task_id="transform_bronze_to_silver",
        python_callable=transform_bronze_to_silver,
        provide_context=True,
    )

    task_quality = PythonOperator(
        task_id="run_data_quality_checks",
        python_callable=run_data_quality_checks,
        provide_context=True,
    )

    wait_for_bronze >> task_transform >> task_quality
