"""
Pipeline orchestrator.

Coordinates the full ML pipeline: ingestion -> preprocessing ->
feature engineering -> training -> evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.data.storage import StockDatabase
from src.data.ingestion import StockIngestion
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


class PipelineOrchestrator:
    """Orchestrates the end-to-end ML pipeline."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)

        db_config = self.config["database"]
        self.db = StockDatabase(
            host=db_config["host"],
            port=db_config["port"],
            dbname=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
        )
        self.ingestion = StockIngestion(db=self.db)
        self.preprocessor = DataPreprocessor()

        # Feature engineer with config params
        feat_config = self.config.get("features", {})
        model_config = self.config.get("model", {})
        self.feature_engineer = FeatureEngineer(
            lag_periods=feat_config.get("lag_periods", [1, 2, 3, 5, 10, 21]),
            rolling_windows=feat_config.get("rolling_windows", [5, 10, 21, 63]),
            forecast_horizon=model_config.get("forecast_horizon", 5),
        )

    def run_ingestion(self) -> dict[str, int]:
        """
        Run the data ingestion step for all configured tickers.

        Returns
        -------
        dict[str, int]
            Mapping of ticker -> rows inserted.
        """
        data_config = self.config["data"]
        tickers = data_config["tickers"]
        start_date = data_config["start_date"]
        end_date = data_config.get("end_date")

        logger.info(f"Starting ingestion for {len(tickers)} tickers...")
        results = self.ingestion.ingest_multiple(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            incremental=True,
        )

        total_rows = sum(v for v in results.values() if v > 0)
        logger.info(f"Ingestion complete. Total new rows: {total_rows}")

        return results

    def run_preprocessing(self, ticker: str):
        """
        Run preprocessing for a single ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol to preprocess.

        Returns
        -------
        pd.DataFrame
            Preprocessed data ready for feature engineering.
        """
        logger.info(f"Preprocessing {ticker}...")
        raw_df = self.db.get_prices(ticker)

        if raw_df.empty:
            logger.warning(f"No data found for {ticker}. Run ingestion first.")
            return raw_df

        processed_df = self.preprocessor.process(raw_df)
        logger.info(f"{ticker}: preprocessed {len(processed_df)} rows.")

        return processed_df

    def run_feature_engineering(self, ticker: str):
        """
        Run feature engineering for a single ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol.

        Returns
        -------
        pd.DataFrame
            DataFrame with all features and target created.
        """
        logger.info(f"Feature engineering for {ticker}...")

        # Get preprocessed data
        processed_df = self.run_preprocessing(ticker)
        if processed_df.empty:
            return processed_df

        # Set date as index for feature engineering
        processed_df = processed_df.set_index("date")

        # Create features
        features_df = self.feature_engineer.create_features(processed_df)
        logger.info(f"{ticker}: {features_df.shape[1]} features, {len(features_df)} rows")

        return features_df

    def run_full_pipeline(self):
        """
        Run the complete pipeline: ingestion -> preprocessing -> features.

        Training and evaluation steps will be added as the project evolves.
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL PIPELINE")
        logger.info("=" * 60)

        # Step 1: Ingestion
        ingestion_results = self.run_ingestion()
        logger.info(f"Ingestion results: {ingestion_results}")

        # Step 2: Preprocessing + Feature Engineering
        tickers = self.config["data"]["tickers"]
        feature_data = {}
        for ticker in tickers:
            feature_data[ticker] = self.run_feature_engineering(ticker)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        return feature_data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    orchestrator = PipelineOrchestrator()
    orchestrator.run_full_pipeline()
