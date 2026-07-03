"""
Data preprocessing module.

Handles cleaning, normalization, and preparation of raw stock data
for feature engineering and model training.
"""

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses raw stock data for ML pipeline."""

    def __init__(self, handle_missing: str = "ffill", remove_outliers: bool = False):
        """
        Parameters
        ----------
        handle_missing : str
            Strategy for missing values: "ffill", "interpolate", or "drop".
        remove_outliers : bool
            Whether to remove statistical outliers from returns.
        """
        self.handle_missing = handle_missing
        self.remove_outliers = remove_outliers

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline for a single ticker's data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw price data with columns: date, open, high, low, close, volume.

        Returns
        -------
        pd.DataFrame
            Cleaned and preprocessed data.
        """
        df = df.copy()

        # Ensure date is datetime and sorted
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Add basic derived columns
        df = self._add_returns(df)

        # Remove outliers if configured
        if self.remove_outliers:
            df = self._remove_outliers(df)

        # Validate data integrity
        self._validate(df)

        logger.info(f"Preprocessing complete. Shape: {df.shape}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configured strategy."""
        numeric_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [col for col in numeric_cols if col in df.columns]

        missing_before = df[available_cols].isnull().sum().sum()

        if self.handle_missing == "ffill":
            df[available_cols] = df[available_cols].ffill()
            # Backfill any remaining NaN at the beginning
            df[available_cols] = df[available_cols].bfill()
        elif self.handle_missing == "interpolate":
            df[available_cols] = df[available_cols].interpolate(method="time")
            df[available_cols] = df[available_cols].bfill()
        elif self.handle_missing == "drop":
            df = df.dropna(subset=available_cols)
        else:
            raise ValueError(f"Unknown missing value strategy: {self.handle_missing}")

        missing_after = df[available_cols].isnull().sum().sum()
        if missing_before > 0:
            logger.info(
                f"Missing values: {missing_before} -> {missing_after} "
                f"(strategy: {self.handle_missing})"
            )

        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log returns and simple returns."""
        df["return_simple"] = df["close"].pct_change()
        df["return_log"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def _remove_outliers(self, df: pd.DataFrame, n_std: float = 4.0) -> pd.DataFrame:
        """Remove rows where returns exceed n standard deviations."""
        if "return_simple" not in df.columns:
            return df

        mean = df["return_simple"].mean()
        std = df["return_simple"].std()
        lower = mean - n_std * std
        upper = mean + n_std * std

        mask = df["return_simple"].between(lower, upper) | df["return_simple"].isna()
        removed = (~mask).sum()

        if removed > 0:
            logger.info(f"Removed {removed} outlier rows (>{n_std} std from mean)")

        return df[mask].reset_index(drop=True)

    @staticmethod
    def _validate(df: pd.DataFrame):
        """Validate data integrity after preprocessing."""
        assert df["date"].is_monotonic_increasing, "Dates are not sorted"
        assert not df["close"].isnull().any(), "Close prices contain NaN after preprocessing"
        assert (df["close"] > 0).all(), "Close prices contain non-positive values"
