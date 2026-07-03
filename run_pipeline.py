"""
Entry point for running the stock prediction pipeline.

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --step ingestion   # Run only ingestion
    python run_pipeline.py --step preprocess  # Run only preprocessing
"""

import argparse
import logging
import sys

from src.pipeline.orchestrator import PipelineOrchestrator


def setup_logging(level: str = "INFO"):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="a", encoding="utf-8"),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Stock Trend Predictor Pipeline")
    parser.add_argument(
        "--step",
        choices=["ingestion", "preprocess", "features", "full"],
        default="full",
        help="Pipeline step to run (default: full)"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"Running pipeline step: {args.step}")

    orchestrator = PipelineOrchestrator(config_path=args.config)

    if args.step == "ingestion":
        results = orchestrator.run_ingestion()
        logger.info(f"Ingestion results: {results}")

    elif args.step == "preprocess":
        tickers = orchestrator.config["data"]["tickers"]
        for ticker in tickers:
            orchestrator.run_preprocessing(ticker)

    elif args.step == "features":
        tickers = orchestrator.config["data"]["tickers"]
        for ticker in tickers:
            orchestrator.run_feature_engineering(ticker)

    elif args.step == "full":
        orchestrator.run_full_pipeline()

    logger.info("Done.")


if __name__ == "__main__":
    main()
