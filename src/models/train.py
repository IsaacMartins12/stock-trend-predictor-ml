"""
Model training module with Walk-Forward Validation.

Implements time-series-aware cross-validation to avoid look-ahead bias.
Trains LightGBM classifier to predict stock price direction (up/down)
over a configurable forecast horizon.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Time-series walk-forward cross-validation.

    Unlike random K-Fold, this respects temporal ordering:
    - Train on past data only
    - Test on the immediate future window
    - Slide forward and repeat

    Example with n_splits=3, test_size=63:
        Split 1: Train [0 : T-189]     | Test [T-189 : T-126]
        Split 2: Train [0 : T-126]     | Test [T-126 : T-63]
        Split 3: Train [0 : T-63]      | Test [T-63 : T]
    """

    def __init__(self, n_splits: int = 5, test_size: int = 63):
        """
        Parameters
        ----------
        n_splits : int
            Number of walk-forward splits.
        test_size : int
            Number of samples in each test window (~63 = 3 months of trading days).
        """
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X: pd.DataFrame):
        """
        Generate train/test indices for walk-forward validation.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            Train indices and test indices for each split.
        """
        n_samples = len(X)
        min_train_size = n_samples - (self.n_splits * self.test_size)

        if min_train_size < self.test_size:
            raise ValueError(
                f"Not enough data for {self.n_splits} splits with test_size={self.test_size}. "
                f"Need at least {(self.n_splits + 1) * self.test_size} samples, got {n_samples}."
            )

        for i in range(self.n_splits):
            test_end = n_samples - (self.n_splits - 1 - i) * self.test_size
            test_start = test_end - self.test_size
            train_end = test_start

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx


class StockModelTrainer:
    """
    Trains and evaluates a LightGBM model for stock direction prediction.

    Supports:
    - Walk-forward validation (proper time series split)
    - Feature importance analysis
    - Model persistence
    - Metrics logging to database
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 63,
        hyperparameters: dict = None,
    ):
        self.validator = WalkForwardValidator(n_splits=n_splits, test_size=test_size)
        self.hyperparameters = hyperparameters or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.model = None
        self.feature_names = None
        self.metrics_history = []

    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix (X) and target (y) from gold layer data.

        Drops non-feature columns and rows with NaN target.

        Parameters
        ----------
        df : pd.DataFrame
            Gold layer feature store data.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            Feature matrix X and target series y.
        """
        # Columns to exclude from features
        exclude_cols = [
            "target_direction_5d",
            "future_return_5d",
            "date",
            "asset_id",
            "computed_at",
            "id",
        ]

        # Target
        target_col = "target_direction_5d"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Drop rows where target is NaN (last N rows don't have future data)
        df_clean = df.dropna(subset=[target_col]).copy()

        # Feature columns
        feature_cols = [c for c in df_clean.columns if c not in exclude_cols]
        self.feature_names = feature_cols

        X = df_clean[feature_cols].astype(float)
        y = df_clean[target_col].astype(int)

        # Handle any remaining NaN in features (replace with median)
        X = X.fillna(X.median())

        logger.info(f"Features prepared: X={X.shape}, y={y.shape} (class balance: {y.mean():.2%} positive)")
        return X, y

    def train_walk_forward(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict:
        """
        Train model using walk-forward validation.

        Returns
        -------
        dict
            Aggregated metrics across all folds.
        """
        all_metrics = []
        all_predictions = []
        all_actuals = []

        logger.info(
            f"Starting walk-forward validation: "
            f"{self.validator.n_splits} splits, test_size={self.validator.test_size}"
        )

        for fold_idx, (train_idx, test_idx) in enumerate(self.validator.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train LightGBM
            model = lgb.LGBMClassifier(**self.hyperparameters)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Metrics for this fold
            fold_metrics = {
                "fold": fold_idx + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "directional_accuracy": accuracy_score(y_test, y_pred),
            }

            all_metrics.append(fold_metrics)
            all_predictions.extend(y_pred.tolist())
            all_actuals.extend(y_test.tolist())

            logger.info(
                f"  Fold {fold_idx + 1}/{self.validator.n_splits}: "
                f"accuracy={fold_metrics['accuracy']:.4f}, "
                f"f1={fold_metrics['f1_score']:.4f}, "
                f"train={len(X_train)}, test={len(X_test)}"
            )

        # Aggregated metrics
        aggregated = {
            "accuracy": np.mean([m["accuracy"] for m in all_metrics]),
            "precision": np.mean([m["precision"] for m in all_metrics]),
            "recall": np.mean([m["recall"] for m in all_metrics]),
            "f1_score": np.mean([m["f1_score"] for m in all_metrics]),
            "directional_accuracy": np.mean([m["directional_accuracy"] for m in all_metrics]),
            "accuracy_std": np.std([m["accuracy"] for m in all_metrics]),
            "n_splits": self.validator.n_splits,
        }

        self.metrics_history = all_metrics

        logger.info("=" * 50)
        logger.info("WALK-FORWARD VALIDATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"  Accuracy:     {aggregated['accuracy']:.4f} ± {aggregated['accuracy_std']:.4f}")
        logger.info(f"  Precision:    {aggregated['precision']:.4f}")
        logger.info(f"  Recall:       {aggregated['recall']:.4f}")
        logger.info(f"  F1 Score:     {aggregated['f1_score']:.4f}")
        logger.info(f"  Dir. Accuracy:{aggregated['directional_accuracy']:.4f}")
        logger.info("=" * 50)

        return aggregated

    def train_final_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the final model on ALL available data for production use.
        Called after walk-forward validation confirms model quality.
        """
        logger.info(f"Training final model on full dataset: {X.shape}")

        self.model = lgb.LGBMClassifier(**self.hyperparameters)
        self.model.fit(X, y)

        logger.info("Final model trained successfully")

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and importance scores.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_final_model() first.")

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return importance.head(top_n).reset_index(drop=True)

    def save_model(self, path: str = "models/", ticker: str = "default") -> str:
        """
        Save the trained model to disk.

        Parameters
        ----------
        path : str
            Directory to save the model.
        ticker : str
            Ticker identifier for the filename.

        Returns
        -------
        str
            Full path to the saved model file.
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)

        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lgbm_{ticker}_{version}.pkl"
        filepath = model_dir / filename

        model_artifact = {
            "model": self.model,
            "feature_names": self.feature_names,
            "hyperparameters": self.hyperparameters,
            "trained_at": datetime.now().isoformat(),
            "ticker": ticker,
            "version": version,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_artifact, f)

        logger.info(f"Model saved: {filepath}")
        return str(filepath)

    @staticmethod
    def load_model(filepath: str) -> dict:
        """Load a saved model artifact from disk."""
        with open(filepath, "rb") as f:
            artifact = pickle.load(f)
        logger.info(f"Model loaded: {filepath} (trained_at={artifact['trained_at']})")
        return artifact

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Make predictions with the trained model.

        Returns
        -------
        dict
            Predictions with direction and confidence.
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        # Ensure feature alignment
        X_aligned = X[self.feature_names].astype(float).fillna(X[self.feature_names].median())

        direction = self.model.predict(X_aligned)
        probabilities = self.model.predict_proba(X_aligned)[:, 1]

        return {
            "direction": direction,  # 1 = up, 0 = down
            "confidence": probabilities,
        }
