"""
DAG: Model Training - Walk-Forward Validation + Final Model

Schedule: Semanal (domingos) ou manual trigger
Responsabilidade: Treinar modelo de ML com dados da Gold layer,
                  avaliar com walk-forward validation, e persistir artefatos.

Pipeline:
    [gold.feature_store] → feature prep → walk-forward CV → final model → save
"""

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)

default_args = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def load_gold_data(**context):
    """
    Load feature store data from gold layer for model training.
    """
    import pandas as pd
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

    ticker_data = {}

    with psycopg.connect(**conn_params) as conn:
        for ticker in tickers:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT fs.*
                    FROM gold.feature_store fs
                    JOIN assets a ON fs.asset_id = a.id
                    WHERE a.ticker = %s
                    ORDER BY fs.date ASC
                    """,
                    (ticker,),
                )
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()

            if rows:
                df = pd.DataFrame(rows, columns=columns)
                # Convert Decimal to float
                for col in df.columns:
                    if col not in ("id", "date", "asset_id", "computed_at"):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df["date"] = pd.to_datetime(df["date"])
                ticker_data[ticker] = len(df)
                logger.info(f"{ticker}: loaded {len(df)} rows from gold.feature_store")
            else:
                ticker_data[ticker] = 0
                logger.warning(f"{ticker}: no data in gold layer")

    context["ti"].xcom_push(key="data_summary", value=ticker_data)
    return ticker_data


def train_and_evaluate(**context):
    """
    Train LightGBM model with walk-forward validation for each ticker.
    Logs metrics to the database and saves model artifacts.
    """
    import pandas as pd
    import numpy as np
    import psycopg
    import yaml
    import json

    from src.models.train import StockModelTrainer

    with open("/opt/airflow/configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tickers = config["data"]["tickers"]
    model_config = config["model"]
    train_config = model_config["train_test_split"]

    conn_params = {
        "host": "postgres",
        "port": 5432,
        "dbname": "stock_predictor",
        "user": "stock_user",
        "password": "stock_pass",
    }

    all_results = {}

    with psycopg.connect(**conn_params) as conn:
        for ticker in tickers:
            logger.info(f"{'='*50}")
            logger.info(f"TRAINING MODEL: {ticker}")
            logger.info(f"{'='*50}")

            try:
                # Load gold data
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT fs.*
                        FROM gold.feature_store fs
                        JOIN assets a ON fs.asset_id = a.id
                        WHERE a.ticker = %s
                        ORDER BY fs.date ASC
                        """,
                        (ticker,),
                    )
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()

                if not rows:
                    logger.warning(f"{ticker}: no gold data, skipping")
                    all_results[ticker] = {"status": "skipped"}
                    continue

                df = pd.DataFrame(rows, columns=columns)
                for col in df.columns:
                    if col not in ("id", "date", "asset_id", "computed_at"):
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # Initialize trainer
                hp_config = model_config["hyperparameters"]["lightgbm"]
                # Merge with LightGBM-specific defaults
                full_hyperparameters = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "random_state": 42,
                    "verbose": -1,
                    "n_jobs": -1,
                    **hp_config,
                }

                trainer = StockModelTrainer(
                    n_splits=train_config["n_splits"],
                    test_size=train_config["test_size"],
                    hyperparameters=full_hyperparameters,
                )

                # Prepare features
                X, y = trainer.prepare_features(df)

                # Walk-forward validation
                metrics = trainer.train_walk_forward(X, y)

                # Train final model on all data
                trainer.train_final_model(X, y)

                # Feature importance
                importance = trainer.get_feature_importance(top_n=15)
                logger.info(f"\nTop 15 features for {ticker}:")
                for _, row in importance.iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']}")

                # Save model artifact
                model_path = trainer.save_model(
                    path="/opt/airflow/models/",
                    ticker=ticker.replace(".", "_"),
                )

                # Generate model version
                model_version = f"lgbm_v1_{datetime.now().strftime('%Y%m%d')}"

                # Log metrics to database
                with conn.cursor() as cur:
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            cur.execute(
                                """
                                INSERT INTO model_metrics
                                    (model_version, ticker, metric_name, metric_value,
                                     n_train_samples, n_test_samples, hyperparameters)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    model_version,
                                    ticker,
                                    metric_name,
                                    float(metric_value),
                                    len(X),
                                    train_config["test_size"],
                                    json.dumps(model_config["hyperparameters"]["lightgbm"]),
                                ),
                            )
                conn.commit()

                all_results[ticker] = {
                    "status": "success",
                    "metrics": {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
                    "model_path": model_path,
                    "model_version": model_version,
                    "n_samples": len(X),
                }

                logger.info(f"✓ {ticker}: model trained and saved ({model_path})")

            except Exception as e:
                conn.rollback()
                logger.error(f"✗ {ticker} training failed: {e}")
                all_results[ticker] = {"status": "failed", "error": str(e)}

    context["ti"].xcom_push(key="training_results", value=all_results)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for ticker, result in all_results.items():
        if result["status"] == "success":
            m = result["metrics"]
            logger.info(
                f"  {ticker}: accuracy={m.get('accuracy', 0):.4f}, "
                f"f1={m.get('f1_score', 0):.4f}"
            )
        else:
            logger.info(f"  {ticker}: {result['status']}")
    logger.info("=" * 60)

    return all_results


def generate_predictions(**context):
    """
    Use the trained model to generate predictions for the most recent data.
    Stores predictions in gold.predictions table.
    """
    import pandas as pd
    import numpy as np
    import psycopg
    import yaml
    import pickle
    from pathlib import Path
    from datetime import timedelta

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

    training_results = context["ti"].xcom_pull(key="training_results")
    prediction_date = context["ds"]

    with psycopg.connect(**conn_params) as conn:
        for ticker in tickers:
            try:
                # Determine model path (XCom or filesystem fallback)
                model_path = None
                model_version = None

                if training_results and training_results.get(ticker, {}).get("status") == "success":
                    model_path = training_results[ticker]["model_path"]
                    model_version = training_results[ticker]["model_version"]
                else:
                    # Fallback: find latest model file on disk
                    ticker_safe = ticker.replace(".", "_")
                    models_dir = Path("/opt/airflow/models/")
                    model_files = sorted(
                        models_dir.glob(f"lgbm_{ticker_safe}_*.pkl"),
                        reverse=True,
                    )
                    if model_files:
                        model_path = str(model_files[0])
                        model_version = model_files[0].stem
                    else:
                        logger.warning(f"{ticker}: no model found, skipping")
                        continue

                # Load the model
                with open(model_path, "rb") as f:
                    artifact = pickle.load(f)

                model = artifact["model"]
                feature_names = artifact["feature_names"]

                # Get latest features from gold
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT fs.*
                        FROM gold.feature_store fs
                        JOIN assets a ON fs.asset_id = a.id
                        WHERE a.ticker = %s
                        ORDER BY fs.date DESC
                        LIMIT 1
                        """,
                        (ticker,),
                    )
                    columns = [desc[0] for desc in cur.description]
                    row = cur.fetchone()

                if not row:
                    continue

                df = pd.DataFrame([row], columns=columns)
                for col in df.columns:
                    if col not in ("id", "date", "asset_id", "computed_at"):
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # Align features
                available_features = [f for f in feature_names if f in df.columns]
                X_pred = df[available_features].astype(float).fillna(0)

                # Predict
                direction = int(model.predict(X_pred)[0])
                confidence = float(model.predict_proba(X_pred)[0][1])

                # Calculate target date
                latest_date = pd.to_datetime(df["date"].iloc[0])
                target_date = (latest_date + timedelta(days=forecast_horizon + 2)).strftime("%Y-%m-%d")

                # Get asset_id
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM assets WHERE ticker = %s", (ticker,))
                    asset_id = cur.fetchone()[0]

                    # Insert prediction
                    cur.execute(
                        """
                        INSERT INTO gold.predictions
                            (asset_id, prediction_date, target_date, model_version,
                             predicted_direction, confidence)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            asset_id,
                            prediction_date,
                            target_date,
                            model_version,
                            direction,
                            confidence,
                        ),
                    )

                conn.commit()
                direction_label = "📈 ALTA" if direction == 1 else "📉 BAIXA"
                logger.info(
                    f"✓ {ticker}: prediction={direction_label}, "
                    f"confidence={confidence:.2%}, target_date={target_date}"
                )

            except Exception as e:
                conn.rollback()
                logger.error(f"✗ {ticker} prediction failed: {e}")


# ============================================
# DAG Definition
# ============================================
with DAG(
    dag_id="model_training",
    default_args=default_args,
    description="Train ML model with walk-forward validation and generate predictions",
    schedule_interval="0 6 * * 0",  # Domingos às 06:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training", "prediction", "lightgbm"],
    max_active_runs=1,
) as dag:

    task_load_data = PythonOperator(
        task_id="load_gold_data",
        python_callable=load_gold_data,
        provide_context=True,
    )

    task_train = PythonOperator(
        task_id="train_and_evaluate",
        python_callable=train_and_evaluate,
        provide_context=True,
    )

    task_predict = PythonOperator(
        task_id="generate_predictions",
        python_callable=generate_predictions,
        provide_context=True,
    )

    task_load_data >> task_train >> task_predict
