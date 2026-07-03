"""
DAG: Gold Layer - Feature Engineering & Feature Store

Schedule: Diariamente às 20:00 (após silver transformation)
Responsabilidade: Calcular features técnicas e populor o feature store (gold).

Arquitetura Medalhão:
    [silver.stock_prices] → feature engineering → [gold.feature_store]
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

def compute_features(**context):
    """
    Calculate technical indicators and features from silver data.
    Populates the gold.feature_store table.
    """
    import pandas as pd
    import numpy as np
    import psycopg
    import yaml

    with open("/opt/airflow/configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tickers = config["data"]["tickers"]
    forecast_horizon = config["model"]["forecast_horizon"]

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
                # Get asset_id
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM assets WHERE ticker = %s", (ticker,))
                    row = cur.fetchone()
                    if not row:
                        logger.warning(f"{ticker}: not found in assets table")
                        results[ticker] = 0
                        continue
                    asset_id = row[0]

                # Read all silver data for feature calculation
                # (features need full history for rolling calculations)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT date, open, high, low, close, adj_close, volume,
                               return_simple, return_log
                        FROM silver.stock_prices
                        WHERE asset_id = %s
                        ORDER BY date ASC
                        """,
                        (asset_id,),
                    )
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()

                if not rows:
                    results[ticker] = 0
                    continue

                df = pd.DataFrame(rows, columns=columns)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

                # Convert Decimal to float
                for col in df.select_dtypes(include=["object"]).columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                numeric_cols = ["open", "high", "low", "close", "adj_close", "volume", "return_simple", "return_log"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # ---- FEATURE CALCULATION ----

                # Moving Averages
                df["sma_5"] = df["close"].rolling(5).mean()
                df["sma_10"] = df["close"].rolling(10).mean()
                df["sma_21"] = df["close"].rolling(21).mean()
                df["sma_63"] = df["close"].rolling(63).mean()
                df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
                df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

                # Volatility
                df["volatility_21d"] = df["return_log"].rolling(21).std() * np.sqrt(252)
                df["volatility_63d"] = df["return_log"].rolling(63).std() * np.sqrt(252)

                # ATR
                high_low = df["high"] - df["low"]
                high_close = (df["high"] - df["close"].shift()).abs()
                low_close = (df["low"] - df["close"].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df["atr_14"] = true_range.rolling(14).mean()

                # RSI
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                avg_gain_14 = gain.rolling(14).mean()
                avg_loss_14 = loss.rolling(14).mean()
                rs_14 = avg_gain_14 / avg_loss_14.replace(0, np.nan)
                df["rsi_14"] = 100 - (100 / (1 + rs_14))

                avg_gain_7 = gain.rolling(7).mean()
                avg_loss_7 = loss.rolling(7).mean()
                rs_7 = avg_gain_7 / avg_loss_7.replace(0, np.nan)
                df["rsi_7"] = 100 - (100 / (1 + rs_7))

                # MACD
                df["macd"] = df["ema_12"] - df["ema_26"]
                df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
                df["macd_histogram"] = df["macd"] - df["macd_signal"]

                # Bollinger Bands
                sma_20 = df["close"].rolling(20).mean()
                std_20 = df["close"].rolling(20).std()
                df["bb_upper"] = sma_20 + (2 * std_20)
                df["bb_lower"] = sma_20 - (2 * std_20)
                df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma_20
                bb_range = df["bb_upper"] - df["bb_lower"]
                df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)

                # Volume features
                df["volume_sma_21"] = df["volume"].rolling(21).mean()
                df["volume_ratio_21"] = df["volume"] / df["volume_sma_21"].replace(0, np.nan)
                df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

                # Return features
                df["return_1d"] = df["close"].pct_change(1)
                df["return_5d"] = df["close"].pct_change(5)
                df["return_21d"] = df["close"].pct_change(21)

                # Lag features
                df["close_lag_1"] = df["close"].shift(1)
                df["close_lag_5"] = df["close"].shift(5)
                df["close_lag_21"] = df["close"].shift(21)

                # Target
                df["future_return_5d"] = df["close"].shift(-forecast_horizon) / df["close"] - 1
                df["target_direction_5d"] = (df["future_return_5d"] > 0).astype(int)

                # Drop rows without enough history for features
                df = df.dropna(subset=["sma_63", "volatility_63d", "rsi_14"])

                # ---- LOAD INTO GOLD ----

                # Only insert rows that don't exist yet
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT MAX(date) FROM gold.feature_store WHERE asset_id = %s",
                        (asset_id,),
                    )
                    last_gold_date = cur.fetchone()[0]

                if last_gold_date:
                    df = df[df.index > pd.to_datetime(last_gold_date)]

                if df.empty:
                    logger.info(f"{ticker}: gold features already up to date")
                    results[ticker] = 0
                    continue

                # Insert features
                feature_cols = [
                    "sma_5", "sma_10", "sma_21", "sma_63", "ema_12", "ema_26",
                    "volatility_21d", "volatility_63d", "atr_14",
                    "rsi_14", "rsi_7", "macd", "macd_signal", "macd_histogram",
                    "bb_upper", "bb_lower", "bb_width", "bb_position",
                    "volume_sma_21", "volume_ratio_21", "obv",
                    "return_1d", "return_5d", "return_21d",
                    "close_lag_1", "close_lag_5", "close_lag_21",
                    "target_direction_5d", "future_return_5d",
                ]

                rows_inserted = 0
                with conn.cursor() as cur:
                    for date_idx, row in df.iterrows():
                        values = [asset_id, date_idx.strftime("%Y-%m-%d")]
                        for col in feature_cols:
                            val = row.get(col)
                            if pd.notna(val):
                                values.append(float(val))
                            else:
                                values.append(None)

                        placeholders = ", ".join(["%s"] * (2 + len(feature_cols)))
                        col_names = ", ".join(["asset_id", "date"] + feature_cols)

                        cur.execute(
                            f"""
                            INSERT INTO gold.feature_store ({col_names})
                            VALUES ({placeholders})
                            ON CONFLICT (asset_id, date) DO UPDATE SET
                                computed_at = NOW()
                            """,
                            values,
                        )
                        rows_inserted += cur.rowcount

                conn.commit()
                results[ticker] = rows_inserted
                logger.info(f"✓ {ticker}: {rows_inserted} rows → gold.feature_store")

            except Exception as e:
                conn.rollback()
                logger.error(f"✗ {ticker} gold feature computation failed: {e}")
                results[ticker] = -1

    context["ti"].xcom_push(key="gold_results", value=results)
    total = sum(v for v in results.values() if v > 0)
    logger.info(f"Gold feature engineering complete | total_rows={total}")
    return results


def validate_feature_store(**context):
    """Validate that feature store has consistent data."""
    import psycopg
    import yaml

    with open("/opt/airflow/configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tickers = config["data"]["tickers"]

    conn_params = {
        "host": "postgres",
        "port": 5432,
        "dbname": "stock_predictor",
        "user": "stock_user",
        "password": "stock_pass",
    }

    with psycopg.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            for ticker in tickers:
                cur.execute(
                    """
                    SELECT COUNT(*), MIN(date), MAX(date)
                    FROM gold.feature_store fs
                    JOIN assets a ON fs.asset_id = a.id
                    WHERE a.ticker = %s
                    """,
                    (ticker,),
                )
                count, min_date, max_date = cur.fetchone()
                logger.info(
                    f"{ticker}: {count} feature rows "
                    f"({min_date} → {max_date})"
                )

    return {"status": "validated"}


# ============================================
# DAG Definition
# ============================================
with DAG(
    dag_id="gold_feature_engineering",
    default_args=default_args,
    description="Compute technical features and populate gold feature store",
    schedule_interval="0 20 * * 1-5",  # 20:00, after silver transformation
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["gold", "features", "medallion", "feature-store"],
    max_active_runs=1,
) as dag:

    # Wait for silver transformation to complete
    wait_for_silver = ExternalTaskSensor(
        task_id="wait_for_silver_transformation",
        external_dag_id="silver_transformation",
        external_task_id="run_data_quality_checks",
        timeout=600,
        poke_interval=30,
        mode="poke",
    )

    task_compute = PythonOperator(
        task_id="compute_features",
        python_callable=compute_features,
        provide_context=True,
    )

    task_validate = PythonOperator(
        task_id="validate_feature_store",
        python_callable=validate_feature_store,
        provide_context=True,
    )

    wait_for_silver >> task_compute >> task_validate
