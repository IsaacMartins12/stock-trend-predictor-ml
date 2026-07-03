"""
Feature engineering module for time series data.
Generates temporal features, technical indicators, and statistical features
for feeding into ML models.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates features from OHLCV price data for time series prediction.
    All features are calculated using only past data to avoid look-ahead bias.
    """

    def __init__(
        self,
        lag_periods: list[int] = None,
        rolling_windows: list[int] = None,
        forecast_horizon: int = 5,
    ):
        self.lag_periods = lag_periods or [1, 2, 3, 5, 10, 21]
        self.rolling_windows = rolling_windows or [5, 10, 21, 63]
        self.forecast_horizon = forecast_horizon

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline completo de criação de features.
        Input: DataFrame com colunas [open, high, low, close, volume]
        Output: DataFrame com todas as features + target
        """
        df = df.copy()

        # Garantir que index é datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Features de retorno
        df = self._add_return_features(df)

        # Features de lag
        df = self._add_lag_features(df)

        # Features de rolling statistics
        df = self._add_rolling_features(df)

        # Indicadores técnicos
        df = self._add_technical_indicators(df)

        # Features de volatilidade
        df = self._add_volatility_features(df)

        # Features temporais (dia da semana, mês, etc.)
        df = self._add_temporal_features(df)

        # Target (variável alvo)
        df = self._add_target(df)

        # Remover rows com NaN gerados pelos lags/rolling
        initial_rows = len(df)
        df = df.dropna()
        logger.info(
            f"Features created: {df.shape[1]} columns, "
            f"{len(df)} rows (dropped {initial_rows - len(df)} NaN rows)"
        )

        return df

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retornos simples e logarítmicos."""
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_21d"] = df["close"].pct_change(21)

        df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))
        df["log_return_5d"] = np.log(df["close"] / df["close"].shift(5))

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag features do preço de fechamento e retorno."""
        for lag in self.lag_periods:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estatísticas de janela móvel (rolling)."""
        for window in self.rolling_windows:
            # Média móvel do preço
            df[f"sma_{window}"] = df["close"].rolling(window).mean()

            # Desvio padrão móvel
            df[f"std_{window}"] = df["close"].rolling(window).std()

            # Posição relativa à média móvel (z-score)
            df[f"zscore_{window}"] = (
                (df["close"] - df[f"sma_{window}"]) / df[f"std_{window}"]
            )

            # Volume médio
            df[f"volume_sma_{window}"] = df["volume"].rolling(window).mean()

            # Razão volume atual vs média
            df[f"volume_ratio_{window}"] = df["volume"] / df[f"volume_sma_{window}"]

            # Min/Max rolling
            df[f"rolling_min_{window}"] = df["close"].rolling(window).min()
            df[f"rolling_max_{window}"] = df["close"].rolling(window).max()

            # Posição dentro do range (0 = no mínimo, 1 = no máximo)
            range_val = df[f"rolling_max_{window}"] - df[f"rolling_min_{window}"]
            df[f"range_position_{window}"] = (
                (df["close"] - df[f"rolling_min_{window}"]) / range_val.replace(0, np.nan)
            )

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicadores técnicos clássicos."""
        # RSI (Relative Strength Index) - 14 períodos
        df["rsi_14"] = self._calculate_rsi(df["close"], 14)
        df["rsi_7"] = self._calculate_rsi(df["close"], 7)

        # MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma_20 + (2 * std_20)
        df["bb_lower"] = sma_20 - (2 * std_20)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma_20
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(14).mean()

        # OBV (On-Balance Volume)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de volatilidade."""
        # Volatilidade realizada (anualizada)
        df["volatility_21d"] = df["log_return_1d"].rolling(21).std() * np.sqrt(252)
        df["volatility_63d"] = df["log_return_1d"].rolling(63).std() * np.sqrt(252)

        # Garman-Klass volatility estimator
        log_hl = np.log(df["high"] / df["low"]) ** 2
        log_co = np.log(df["close"] / df["open"]) ** 2
        df["gk_volatility"] = np.sqrt(
            (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(21).mean() * 252
        )

        # Variação intraday
        df["intraday_range"] = (df["high"] - df["low"]) / df["open"]

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de calendário/sazonalidade."""
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["is_month_start"] = df.index.is_month_start.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria a variável target: direção do preço nos próximos N dias.
        target = 1 se preço subiu, 0 se caiu.
        """
        # Retorno futuro (para classificação de direção)
        df["future_return"] = df["close"].shift(-self.forecast_horizon) / df["close"] - 1

        # Target binário: 1 = alta, 0 = baixa
        df["target"] = (df["future_return"] > 0).astype(int)

        return df

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        """Calcula RSI (Relative Strength Index)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi
