"""
Data ingestion module.

Responsible for fetching stock market data from external sources (yfinance)
and persisting it into PostgreSQL via the storage layer.

Supports:
- Full historical load (backfill)
- Incremental updates (only fetch new data since last stored date)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from src.data.storage import StockDatabase

logger = logging.getLogger(__name__)


class StockIngestion:
    """Handles extraction and loading of stock market data."""

    def __init__(self, db: StockDatabase):
        self.db = db

    def ingest_ticker(
        self,
        ticker: str,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        incremental: bool = True,
    ) -> int:
        """
        Ingest OHLCV data for a single ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g., "PETR4.SA").
        start_date : str
            Earliest date to fetch if no data exists.
        end_date : str, optional
            End date. Defaults to today.
        incremental : bool
            If True, only fetches data after the last stored date.

        Returns
        -------
        int
            Number of new rows inserted.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Ensure asset exists in database
        asset_id = self.db.upsert_asset(ticker)

        # Determine fetch window
        if incremental:
            last_date = self.db.get_last_date(ticker)
            if last_date:
                # Start one day after last stored date
                fetch_start = (
                    pd.to_datetime(last_date) + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                if fetch_start >= end_date:
                    logger.info(f"{ticker}: already up to date (last={last_date})")
                    return 0
            else:
                fetch_start = start_date
        else:
            fetch_start = start_date

        logger.info(f"Fetching {ticker}: {fetch_start} → {end_date}")

        # Extract from yfinance
        df = self._fetch_from_yfinance(ticker, fetch_start, end_date)

        if df.empty:
            logger.warning(f"No data returned for {ticker} ({fetch_start} to {end_date})")
            return 0

        # Load into PostgreSQL
        rows_inserted = self.db.insert_prices(asset_id, df)

        logger.info(
            f"✓ {ticker}: {rows_inserted} rows inserted "
            f"({fetch_start} → {end_date})"
        )
        return rows_inserted

    def ingest_multiple(
        self,
        tickers: list[str],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        incremental: bool = True,
    ) -> dict[str, int]:
        """
        Ingest data for multiple tickers.

        Returns
        -------
        dict
            Mapping of ticker -> rows inserted.
        """
        results = {}
        for ticker in tickers:
            try:
                rows = self.ingest_ticker(
                    ticker,
                    start_date=start_date,
                    end_date=end_date,
                    incremental=incremental,
                )
                results[ticker] = rows
            except Exception as e:
                logger.error(f"Failed to ingest {ticker}: {e}")
                results[ticker] = -1

        total = sum(v for v in results.values() if v > 0)
        logger.info(f"Ingestion complete: {total} total rows across {len(tickers)} tickers")
        return results

    @staticmethod
    def _fetch_from_yfinance(
        ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Returns a clean DataFrame with standardized column names.
        """
        raw = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )

        if raw.empty:
            return pd.DataFrame()

        # Handle multi-level columns from yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index()

        # Standardize column names
        column_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=column_map)

        # Ensure date is string format for DB insertion
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # Drop rows with all NaN prices
        price_cols = ["open", "high", "low", "close"]
        df = df.dropna(subset=price_cols, how="all")

        return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]
