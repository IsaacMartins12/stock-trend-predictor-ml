"""
FastAPI Model Serving Application.

Exposes REST endpoints for:
- Real-time stock predictions
- Model performance metrics
- Model monitoring & drift detection
- Health checks
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ============================================
# App Configuration
# ============================================
app = FastAPI(
    title="Stock Trend Predictor API",
    description="ML-powered stock direction prediction service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Database connection params (from environment in Docker)
import os

DB_PARAMS = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "stock_predictor"),
    "user": os.getenv("POSTGRES_USER", "stock_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "stock_pass"),
}

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models/"))


# ============================================
# Response Models
# ============================================
class PredictionResponse(BaseModel):
    ticker: str
    prediction: str  # "UP" or "DOWN"
    confidence: float
    target_date: str
    model_version: str
    predicted_at: str


class HealthResponse(BaseModel):
    status: str
    database: str
    models_loaded: int
    timestamp: str


class MetricsResponse(BaseModel):
    ticker: str
    model_version: str
    accuracy: Optional[float]
    f1_score: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    directional_accuracy: Optional[float]
    n_train_samples: Optional[int]
    trained_at: Optional[str]


class MonitorResponse(BaseModel):
    ticker: str
    model_age_days: int
    is_stale: bool
    drift_detected: bool
    drifted_features: int
    total_features: int
    recommendation: str
    last_prediction_accuracy: Optional[float]


# ============================================
# Helper Functions
# ============================================
def get_db_connection():
    """Get a database connection."""
    return psycopg.connect(**DB_PARAMS)


def load_latest_model(ticker: str) -> dict:
    """Load the most recent model artifact for a ticker."""
    ticker_safe = ticker.replace(".", "_")
    model_files = sorted(
        MODELS_DIR.glob(f"lgbm_{ticker_safe}_*.pkl"),
        reverse=True,
    )
    if not model_files:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for {ticker}. Run model training first.",
        )

    with open(model_files[0], "rb") as f:
        artifact = pickle.load(f)

    return artifact


# ============================================
# Endpoints
# ============================================
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Service health check - verifies database and model availability."""
    # Check database
    db_status = "healthy"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception:
        db_status = "unhealthy"

    # Check models
    model_count = len(list(MODELS_DIR.glob("lgbm_*.pkl")))

    status = "healthy" if db_status == "healthy" and model_count > 0 else "degraded"

    return HealthResponse(
        status=status,
        database=db_status,
        models_loaded=model_count,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict(ticker: str):
    """
    Get the latest prediction for a stock ticker.

    Loads the most recent model and latest features from the gold layer,
    then generates a real-time prediction.
    """
    ticker = ticker.upper()

    # Load model
    artifact = load_latest_model(ticker)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    model_version = artifact.get("version", "unknown")

    # Get latest features from gold layer
    try:
        with get_db_connection() as conn:
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
            raise HTTPException(
                status_code=404,
                detail=f"No feature data found for {ticker} in gold layer.",
            )

        df = pd.DataFrame([row], columns=columns)
        for col in df.columns:
            if col not in ("id", "date", "asset_id", "computed_at"):
                df[col] = pd.to_numeric(df[col], errors="coerce")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Align features and predict
    available_features = [f for f in feature_names if f in df.columns]
    X_pred = df[available_features].astype(float).fillna(0)

    direction = int(model.predict(X_pred)[0])
    confidence = float(model.predict_proba(X_pred)[0][1])

    # Calculate target date (5 business days ahead)
    latest_date = pd.to_datetime(df["date"].iloc[0])
    target_date = (latest_date + pd.offsets.BDay(5)).strftime("%Y-%m-%d")

    return PredictionResponse(
        ticker=ticker,
        prediction="UP" if direction == 1 else "DOWN",
        confidence=round(confidence if direction == 1 else 1 - confidence, 4),
        target_date=target_date,
        model_version=model_version,
        predicted_at=datetime.now().isoformat(),
    )


@app.get("/model/metrics/{ticker}", response_model=MetricsResponse)
def model_metrics(ticker: str):
    """Get the latest training metrics for a ticker's model."""
    ticker = ticker.upper()

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get latest metrics
                cur.execute(
                    """
                    SELECT model_version, metric_name, metric_value,
                           n_train_samples, trained_at
                    FROM model_metrics
                    WHERE ticker = %s
                    ORDER BY trained_at DESC
                    LIMIT 10
                    """,
                    (ticker,),
                )
                rows = cur.fetchall()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics found for {ticker}. Train the model first.",
            )

        # Parse metrics
        metrics = {}
        model_version = rows[0][0]
        n_train = rows[0][3]
        trained_at = str(rows[0][4]) if rows[0][4] else None

        for row in rows:
            metrics[row[1]] = float(row[2]) if row[2] else None

        return MetricsResponse(
            ticker=ticker,
            model_version=model_version,
            accuracy=metrics.get("accuracy"),
            f1_score=metrics.get("f1_score"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            directional_accuracy=metrics.get("directional_accuracy"),
            n_train_samples=n_train,
            trained_at=trained_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/monitor/{ticker}", response_model=MonitorResponse)
def monitor(ticker: str):
    """
    Model monitoring endpoint.

    Checks:
    - Model age (staleness)
    - Feature drift (PSI-based)
    - Recent prediction accuracy
    """
    ticker = ticker.upper()

    artifact = load_latest_model(ticker)
    trained_at = datetime.fromisoformat(artifact.get("trained_at", datetime.now().isoformat()))
    model_age = (datetime.now() - trained_at).days

    # Check drift using recent vs historical features
    drift_detected = False
    drifted_features = 0
    total_features = 0
    recommendation = "NO_ACTION_NEEDED"
    last_accuracy = None

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get feature counts for drift estimation
                cur.execute(
                    """
                    SELECT COUNT(*) as total,
                           MIN(date) as first_date,
                           MAX(date) as last_date
                    FROM gold.feature_store fs
                    JOIN assets a ON fs.asset_id = a.id
                    WHERE a.ticker = %s
                    """,
                    (ticker,),
                )
                row = cur.fetchone()
                total_features = len(artifact.get("feature_names", []))

                # Check recent predictions accuracy
                cur.execute(
                    """
                    SELECT predicted_direction, actual_direction
                    FROM gold.predictions
                    WHERE asset_id = (SELECT id FROM assets WHERE ticker = %s)
                    AND actual_direction IS NOT NULL
                    ORDER BY target_date DESC
                    LIMIT 20
                    """,
                    (ticker,),
                )
                pred_rows = cur.fetchall()

                if pred_rows:
                    correct = sum(1 for r in pred_rows if r[0] == r[1])
                    last_accuracy = round(correct / len(pred_rows), 4)

    except Exception as e:
        logger.error(f"Monitoring error for {ticker}: {e}")

    # Determine recommendation
    is_stale = model_age > 30
    if is_stale:
        recommendation = "RETRAIN_STALE_MODEL"
    elif last_accuracy is not None and last_accuracy < 0.5:
        recommendation = "RETRAIN_LOW_ACCURACY"
    elif model_age > 14:
        recommendation = "MONITOR_CLOSELY"

    return MonitorResponse(
        ticker=ticker,
        model_age_days=model_age,
        is_stale=is_stale,
        drift_detected=drift_detected,
        drifted_features=drifted_features,
        total_features=total_features,
        recommendation=recommendation,
        last_prediction_accuracy=last_accuracy,
    )


@app.get("/predictions/{ticker}")
def get_predictions_history(ticker: str, limit: int = 10):
    """Get prediction history for a ticker."""
    ticker = ticker.upper()

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.prediction_date, p.target_date, p.model_version,
                           p.predicted_direction, p.confidence,
                           p.actual_direction, p.is_correct
                    FROM gold.predictions p
                    JOIN assets a ON p.asset_id = a.id
                    WHERE a.ticker = %s
                    ORDER BY p.created_at DESC
                    LIMIT %s
                    """,
                    (ticker, limit),
                )
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()

        if not rows:
            return {"ticker": ticker, "predictions": []}

        predictions = []
        for row in rows:
            pred = dict(zip(columns, row))
            pred["prediction_date"] = str(pred["prediction_date"])
            pred["target_date"] = str(pred["target_date"])
            pred["direction_label"] = "UP" if pred["predicted_direction"] == 1 else "DOWN"
            pred["confidence"] = float(pred["confidence"]) if pred["confidence"] else None
            predictions.append(pred)

        return {"ticker": ticker, "predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ============================================
# Startup
# ============================================
@app.on_event("startup")
async def startup_event():
    """Log available models on startup."""
    model_files = list(MODELS_DIR.glob("lgbm_*.pkl"))
    logger.info(f"API started with {len(model_files)} model(s) available")
    for mf in model_files:
        logger.info(f"  - {mf.name}")
