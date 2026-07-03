"""
CLI script to run the data ingestion pipeline.

Usage:
    python run_ingestion.py                     # Incremental update (default tickers)
    python run_ingestion.py --full              # Full historical backfill
    python run_ingestion.py --tickers PETR4.SA VALE3.SA
    python run_ingestion.py --start 2020-01-01
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.data.storage import StockDatabase
from src.data.ingestion import StockIngestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        logger.error("Config file not found: configs/config.yaml")
        sys.exit(1)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Stock data ingestion pipeline")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="List of tickers to ingest (overrides config)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Defaults to config value.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full backfill (ignore incremental logic)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config()

    # Database connection
    db_config = config["database"]
    db = StockDatabase(
        host=db_config["host"],
        port=db_config["port"],
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
    )

    # Health check
    if not db.health_check():
        logger.error(
            "Cannot connect to PostgreSQL. "
            "Make sure the database is running: docker compose up -d"
        )
        sys.exit(1)

    logger.info("✓ Database connection OK")

    # Resolve parameters
    tickers = args.tickers or config["data"]["tickers"]
    start_date = args.start or config["data"]["start_date"]
    end_date = args.end  # None = today

    # Run ingestion
    ingestion = StockIngestion(db)
    results = ingestion.ingest_multiple(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        incremental=not args.full,
    )

    # Summary
    logger.info("=" * 50)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 50)
    for ticker, rows in results.items():
        status = f"{rows} rows" if rows >= 0 else "FAILED"
        logger.info(f"  {ticker}: {status}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
