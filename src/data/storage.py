"""
Database storage module for stock market data.

Uses PostgreSQL (via psycopg v3) for production-grade persistence.
The database is provisioned via Docker Compose with the schema
defined in infrastructure/init.sql.
"""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Optional

import psycopg
from psycopg.rows import tuple_row
import pandas as pd

logger = logging.getLogger(__name__)


class StockDatabase:
    """Manages PostgreSQL connection and operations for stock market data."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
    ):
        self.connection_params = {
            "host": host or os.getenv("POSTGRES_HOST", "localhost"),
            "port": port or int(os.getenv("POSTGRES_PORT", "5432")),
            "dbname": dbname or os.getenv("POSTGRES_DB", "stock_predictor"),
            "user": user or os.getenv("POSTGRES_USER", "stock_user"),
            "password": password or os.getenv("POSTGRES_PASSWORD", "stock_pass"),
        }

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with auto-commit control."""
        conn = psycopg.connect(**self.connection_params, row_factory=tuple_row)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Assets (tickers)
    # ------------------------------------------------------------------

    def upsert_asset(
        self, ticker: str, name: str = None, sector: str = None
    ) -> int:
        """
        Insert or retrieve an asset, returning its database ID.

        Uses ON CONFLICT to handle duplicates gracefully.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO assets (ticker, name, sector)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (ticker) DO UPDATE
                        SET name = COALESCE(EXCLUDED.name, assets.name),
                            sector = COALESCE(EXCLUDED.sector, assets.sector)
                    RETURNING id
                    """,
                    (ticker, name, sector),
                )
                asset_id = cur.fetchone()[0]
        logger.info(f"Upserted asset: {ticker} (id={asset_id})")
        return asset_id

    def get_asset_id(self, ticker: str) -> Optional[int]:
        """Get the database ID for a ticker symbol."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM assets WHERE ticker = %s", (ticker,))
                row = cur.fetchone()
                return row[0] if row else None

    def get_all_tickers(self) -> list[str]:
        """Return all ticker symbols registered in the database."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ticker FROM assets ORDER BY ticker")
                return [row[0] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Stock Prices (OHLCV)
    # ------------------------------------------------------------------

    def insert_prices(self, asset_id: int, df: pd.DataFrame) -> int:
        """
        Bulk insert OHLCV price data using PostgreSQL's execute_values
        for high performance. Skips rows that conflict on (asset_id, date).

        Parameters
        ----------
        asset_id : int
            Database ID of the asset.
        df : pd.DataFrame
            DataFrame with columns: date, open, high, low, close, adj_close, volume.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if df.empty:
            return 0

        required_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        records = df[required_cols].values.tolist()
        values = [(asset_id, *record) for record in records]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO stock_prices
                        (asset_id, date, open, high, low, close, adj_close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (asset_id, date) DO NOTHING
                    """,
                    values,
                )
                rows_inserted = cur.rowcount

        logger.info(f"Inserted {rows_inserted} price rows for asset_id={asset_id}")
        return rows_inserted

    def get_prices(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data for a ticker with optional date filtering.

        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g., "PETR4.SA").
        start_date : str, optional
            Start date filter (YYYY-MM-DD).
        end_date : str, optional
            End date filter (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            DataFrame with OHLCV data sorted by date ascending.
        """
        query = """
            SELECT sp.date, sp.open, sp.high, sp.low, sp.close, sp.adj_close, sp.volume
            FROM stock_prices sp
            JOIN assets a ON sp.asset_id = a.id
            WHERE a.ticker = %s
        """
        params: list = [ticker]

        if start_date:
            query += " AND sp.date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND sp.date <= %s"
            params.append(end_date)

        query += " ORDER BY sp.date ASC"

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()

        df = pd.DataFrame(rows, columns=columns)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])

        return df

    def get_last_date(self, ticker: str) -> Optional[str]:
        """Get the most recent date stored for a given ticker."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT MAX(sp.date)::TEXT
                    FROM stock_prices sp
                    JOIN assets a ON sp.asset_id = a.id
                    WHERE a.ticker = %s
                    """,
                    (ticker,),
                )
                result = cur.fetchone()[0]
        return result

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def insert_features(self, asset_id: int, df: pd.DataFrame) -> int:
        """
        Bulk insert computed features.

        Parameters
        ----------
        asset_id : int
            Database ID of the asset.
        df : pd.DataFrame
            DataFrame with columns: date, feature_name, feature_value.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if df.empty:
            return 0

        records = df[["date", "feature_name", "feature_value"]].values.tolist()
        values = [(asset_id, *record) for record in records]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO features (asset_id, date, feature_name, feature_value)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (asset_id, date, feature_name)
                    DO UPDATE SET feature_value = EXCLUDED.feature_value
                    """,
                    values,
                )
                rows_inserted = cur.rowcount

        logger.info(f"Inserted {rows_inserted} feature rows for asset_id={asset_id}")
        return rows_inserted

    def get_features(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Retrieve features as a pivoted DataFrame (one column per feature).

        Returns
        -------
        pd.DataFrame
            Pivoted features with date as index.
        """
        query = """
            SELECT f.date, f.feature_name, f.feature_value
            FROM features f
            JOIN assets a ON f.asset_id = a.id
            WHERE a.ticker = %s
        """
        params: list = [ticker]

        if start_date:
            query += " AND f.date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND f.date <= %s"
            params.append(end_date)

        query += " ORDER BY f.date ASC"

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()

        df = pd.DataFrame(rows, columns=columns)

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])

        # Pivot: each feature_name becomes a column
        df_pivot = df.pivot_table(
            index="date", columns="feature_name", values="feature_value"
        ).reset_index()
        df_pivot.columns.name = None

        return df_pivot

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def insert_prediction(
        self,
        asset_id: int,
        prediction_date: str,
        target_date: str,
        model_version: str,
        predicted_direction: int = None,
        predicted_return: float = None,
        confidence: float = None,
    ) -> int:
        """Insert a single model prediction."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions
                        (asset_id, prediction_date, target_date, model_version,
                         predicted_direction, predicted_return, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        asset_id,
                        prediction_date,
                        target_date,
                        model_version,
                        predicted_direction,
                        predicted_return,
                        confidence,
                    ),
                )
                return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Model Metrics
    # ------------------------------------------------------------------

    def insert_model_metrics(
        self,
        model_version: str,
        ticker: str,
        metrics: dict,
        n_train: int,
        n_test: int,
        hyperparameters: dict = None,
    ):
        """
        Store evaluation metrics for a trained model.

        Parameters
        ----------
        model_version : str
            Identifier for the model version (e.g., "lgbm_v1_2025-07-03").
        ticker : str
            Ticker the model was trained on.
        metrics : dict
            Dict of metric_name -> metric_value.
        n_train : int
            Number of training samples used.
        n_test : int
            Number of test samples used.
        hyperparameters : dict, optional
            Model hyperparameters to log.
        """
        import json

        hp_json = json.dumps(hyperparameters) if hyperparameters else None

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for metric_name, metric_value in metrics.items():
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
                            metric_value,
                            n_train,
                            n_test,
                            hp_json,
                        ),
                    )

        logger.info(
            f"Logged {len(metrics)} metrics for model={model_version}, ticker={ticker}"
        )

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Verify database connectivity."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
