"""
DAG: Historical Backfill - Full Pipeline

Schedule: Manual trigger only (não roda em schedule)
Responsabilidade: Carga inicial completa de dados históricos
                  através de todas as camadas (Bronze → Silver → Gold).

Usar para:
    - Setup inicial do projeto
    - Adicionar novos tickers
    - Reprocessamento completo
"""

import uuid
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
}


def backfill_bronze(**context):
    """Full historical extraction into bronze layer."""
    import pandas as pd
    import yfinance as yf
    import psycopg
    import yaml

    with open("/opt/airflow/configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tickers = config["data"]["tickers"]
    start_date = config["data"]["start_date"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    batch_id = str(uuid.uuid4())

    logger.info(f"Starting full backfill | {start_date} → {end_date} | batch={batch_id}")

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
                logger.info(f"Downloading full history for {ticker}...")
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

                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                df = raw.reset_index()
                df.columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

                # Bulk insert into bronze
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

                    # Log
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
                logger.info(f"✓ {ticker}: {rows_inserted} rows → bronze (backfill)")

            except Exception as e:
                conn.rollback()
                logger.error(f"✗ {ticker} backfill failed: {e}")
                results[ticker] = -1

    context["ti"].xcom_push(key="backfill_results", value=results)
    return results


def backfill_silver(**context):
    """Process all bronze data into silver (same logic as daily DAG)."""
    import pandas as pd
    import numpy as np
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

    results = {}

    with psycopg.connect(**conn_params) as conn:
        for ticker in tickers:
            try:
                # Read ALL bronze data for this ticker
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT date, open, high, low, close, adj_close, volume,
                               ingestion_batch_id
                        FROM bronze.raw_stock_prices
                        WHERE ticker = %s
                        ORDER BY date ASC
                        """,
                        (ticker,),
                    )
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()

                if not rows:
                    results[ticker] = 0
                    continue

                df = pd.DataFrame(rows, columns=columns)
                source_batch = df["ingestion_batch_id"].iloc[-1]
                df = df.drop(columns=["ingestion_batch_id"])

                # Transformations
                df = df.drop_duplicates(subset=["date"], keep="last")
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)

                # Convert Decimal to float (PostgreSQL NUMERIC comes as Decimal)
                numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Quality scoring
                price_cols = ["open", "high", "low", "close"]
                df["quality_score"] = 1.0
                missing_mask = df[price_cols].isnull().any(axis=1)
                df.loc[missing_mask, "quality_score"] -= 0.3
                df.loc[df["volume"] == 0, "quality_score"] -= 0.2

                # Imputation
                df["is_imputed"] = df[price_cols].isnull().any(axis=1)
                df[price_cols] = df[price_cols].ffill()
                df["volume"] = df["volume"].fillna(0)

                # OHLC fix
                invalid = df["high"] < df["low"]
                if invalid.any():
                    df.loc[invalid, ["high", "low"]] = df.loc[invalid, ["low", "high"]].values
                    df.loc[invalid, "quality_score"] -= 0.2

                # Returns
                df["return_simple"] = df["close"].pct_change()
                df["return_log"] = np.log(df["close"] / df["close"].shift(1))
                df = df.dropna(subset=["close"])
                df["quality_score"] = df["quality_score"].clip(0, 1)

                # Ensure asset
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO assets (ticker) VALUES (%s) ON CONFLICT (ticker) DO NOTHING",
                        (ticker,),
                    )
                    cur.execute("SELECT id FROM assets WHERE ticker = %s", (ticker,))
                    asset_id = cur.fetchone()[0]

                    # Truncate and reload for this asset
                    cur.execute(
                        "DELETE FROM silver.stock_prices WHERE asset_id = %s",
                        (asset_id,),
                    )

                    rows_inserted = 0
                    for _, row in df.iterrows():
                        cur.execute(
                            """
                            INSERT INTO silver.stock_prices
                                (asset_id, date, open, high, low, close, adj_close,
                                 volume, return_simple, return_log, is_imputed,
                                 quality_score, source_batch_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        rows_inserted += 1

                conn.commit()
                results[ticker] = rows_inserted
                logger.info(f"✓ {ticker}: {rows_inserted} rows → silver (backfill)")

            except Exception as e:
                conn.rollback()
                logger.error(f"✗ {ticker} silver backfill failed: {e}")
                results[ticker] = -1

    context["ti"].xcom_push(key="silver_backfill_results", value=results)
    return results


def backfill_gold(**context):
    """Compute all features for gold layer from silver data."""
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
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM assets WHERE ticker = %s", (ticker,))
                    row = cur.fetchone()
                    if not row:
                        continue
                    asset_id = row[0]

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
                numeric_cols = ["open", "high", "low", "close", "adj_close", "volume", "return_simple", "return_log"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # Features (same as daily gold DAG)
                df["sma_5"] = df["close"].rolling(5).mean()
                df["sma_10"] = df["close"].rolling(10).mean()
                df["sma_21"] = df["close"].rolling(21).mean()
                df["sma_63"] = df["close"].rolling(63).mean()
                df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
                df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

                df["volatility_21d"] = df["return_log"].rolling(21).std() * np.sqrt(252)
                df["volatility_63d"] = df["return_log"].rolling(63).std() * np.sqrt(252)

                high_low = df["high"] - df["low"]
                high_close = (df["high"] - df["close"].shift()).abs()
                low_close = (df["low"] - df["close"].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df["atr_14"] = true_range.rolling(14).mean()

                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                df["rsi_14"] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)))
                df["rsi_7"] = 100 - (100 / (1 + gain.rolling(7).mean() / loss.rolling(7).mean().replace(0, np.nan)))

                df["macd"] = df["ema_12"] - df["ema_26"]
                df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
                df["macd_histogram"] = df["macd"] - df["macd_signal"]

                sma_20 = df["close"].rolling(20).mean()
                std_20 = df["close"].rolling(20).std()
                df["bb_upper"] = sma_20 + (2 * std_20)
                df["bb_lower"] = sma_20 - (2 * std_20)
                df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma_20
                bb_range = df["bb_upper"] - df["bb_lower"]
                df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)

                df["volume_sma_21"] = df["volume"].rolling(21).mean()
                df["volume_ratio_21"] = df["volume"] / df["volume_sma_21"].replace(0, np.nan)
                df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

                df["return_1d"] = df["close"].pct_change(1)
                df["return_5d"] = df["close"].pct_change(5)
                df["return_21d"] = df["close"].pct_change(21)

                df["close_lag_1"] = df["close"].shift(1)
                df["close_lag_5"] = df["close"].shift(5)
                df["close_lag_21"] = df["close"].shift(21)

                df["future_return_5d"] = df["close"].shift(-forecast_horizon) / df["close"] - 1
                df["target_direction_5d"] = (df["future_return_5d"] > 0).astype(int)

                # Drop NaN from rolling
                df = df.dropna(subset=["sma_63", "volatility_63d", "rsi_14"])

                # Truncate gold for this asset and reload
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

                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM gold.feature_store WHERE asset_id = %s",
                        (asset_id,),
                    )

                    rows_inserted = 0
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
                            f"INSERT INTO gold.feature_store ({col_names}) VALUES ({placeholders})",
                            values,
                        )
                        rows_inserted += 1

                conn.commit()
                results[ticker] = rows_inserted
                logger.info(f"✓ {ticker}: {rows_inserted} rows → gold (backfill)")

            except Exception as e:
                conn.rollback()
                logger.error(f"✗ {ticker} gold backfill failed: {e}")
                results[ticker] = -1

    total = sum(v for v in results.values() if v > 0)
    logger.info(f"Gold backfill complete | total_rows={total}")
    return results


# ============================================
# DAG Definition
# ============================================
with DAG(
    dag_id="backfill_historical",
    default_args=default_args,
    description="Full historical backfill through all medallion layers",
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["backfill", "medallion", "bronze", "silver", "gold"],
    max_active_runs=1,
) as dag:

    task_bronze = PythonOperator(
        task_id="backfill_bronze",
        python_callable=backfill_bronze,
        provide_context=True,
    )

    task_silver = PythonOperator(
        task_id="backfill_silver",
        python_callable=backfill_silver,
        provide_context=True,
    )

    task_gold = PythonOperator(
        task_id="backfill_gold",
        python_callable=backfill_gold,
        provide_context=True,
    )

    task_bronze >> task_silver >> task_gold
