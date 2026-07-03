"""
Model Monitoring & Drift Detection Module.

Monitors model performance over time and detects:
- Feature drift (PSI - Population Stability Index)
- Distribution shifts (Kolmogorov-Smirnov test)
- Prediction accuracy degradation
- Model staleness

Alerts when retraining is recommended.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects feature and prediction drift using statistical tests.

    Methods:
    - PSI (Population Stability Index): measures distribution shift
    - KS Test (Kolmogorov-Smirnov): non-parametric distribution comparison
    - Accuracy tracking: rolling window performance monitoring
    """

    PSI_THRESHOLD_WARNING = 0.1   # Minor drift
    PSI_THRESHOLD_CRITICAL = 0.25  # Significant drift → retrain
    KS_PVALUE_THRESHOLD = 0.05    # Reject null hypothesis of same distribution

    def __init__(self, reference_data: pd.DataFrame = None):
        """
        Parameters
        ----------
        reference_data : pd.DataFrame, optional
            Training data distribution used as baseline for drift detection.
        """
        self.reference_data = reference_data
        self.drift_report = {}

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures the shift between two distributions.
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.25: Moderate shift (monitor)
        - PSI ≥ 0.25: Significant shift (retrain)

        Parameters
        ----------
        expected : np.ndarray
            Reference distribution (training data).
        actual : np.ndarray
            Current distribution (production data).
        bins : int
            Number of bins for discretization.

        Returns
        -------
        float
            PSI value.
        """
        # Remove NaN
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create bins from expected distribution
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            bins + 1,
        )

        # Calculate proportions
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Normalize to proportions
        expected_pct = (expected_counts + 1) / (len(expected) + bins)  # Laplace smoothing
        actual_pct = (actual_counts + 1) / (len(actual) + bins)

        # PSI formula
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return float(psi)

    @staticmethod
    def ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
        """
        Kolmogorov-Smirnov test for distribution comparison.

        Tests H0: both samples come from the same distribution.

        Returns
        -------
        dict
            statistic, p_value, and whether drift is detected.
        """
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) < 10 or len(current) < 10:
            return {"statistic": 0, "p_value": 1.0, "drift_detected": False}

        stat, p_value = stats.ks_2samp(reference, current)

        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": p_value < DriftDetector.KS_PVALUE_THRESHOLD,
        }

    def detect_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: list[str] = None,
    ) -> dict:
        """
        Detect drift across multiple features.

        Parameters
        ----------
        reference_df : pd.DataFrame
            Historical/training data.
        current_df : pd.DataFrame
            Recent/production data.
        feature_columns : list[str], optional
            Columns to check. If None, checks all numeric columns.

        Returns
        -------
        dict
            Drift report per feature with PSI and KS results.
        """
        if feature_columns is None:
            feature_columns = reference_df.select_dtypes(include=[np.number]).columns.tolist()

        report = {}
        drifted_features = []

        for col in feature_columns:
            if col not in reference_df.columns or col not in current_df.columns:
                continue

            ref_values = reference_df[col].values.astype(float)
            cur_values = current_df[col].values.astype(float)

            psi = self.calculate_psi(ref_values, cur_values)
            ks = self.ks_test(ref_values, cur_values)

            # Determine drift severity
            if psi >= self.PSI_THRESHOLD_CRITICAL:
                severity = "CRITICAL"
                drifted_features.append(col)
            elif psi >= self.PSI_THRESHOLD_WARNING:
                severity = "WARNING"
                drifted_features.append(col)
            else:
                severity = "OK"

            report[col] = {
                "psi": round(psi, 4),
                "ks_statistic": round(ks["statistic"], 4),
                "ks_p_value": round(ks["p_value"], 4),
                "severity": severity,
            }

        # Summary
        n_drifted = len(drifted_features)
        n_total = len(feature_columns)
        drift_ratio = n_drifted / n_total if n_total > 0 else 0

        self.drift_report = {
            "features": report,
            "summary": {
                "total_features": n_total,
                "drifted_features": n_drifted,
                "drift_ratio": round(drift_ratio, 4),
                "drifted_feature_names": drifted_features,
                "recommendation": self._get_recommendation(drift_ratio, n_drifted),
                "checked_at": datetime.now().isoformat(),
            },
        }

        logger.info(
            f"Drift detection: {n_drifted}/{n_total} features drifted "
            f"({drift_ratio:.1%}) → {self.drift_report['summary']['recommendation']}"
        )

        return self.drift_report

    @staticmethod
    def _get_recommendation(drift_ratio: float, n_drifted: int) -> str:
        """Generate recommendation based on drift severity."""
        if drift_ratio >= 0.3:
            return "RETRAIN_IMMEDIATELY"
        elif drift_ratio >= 0.15 or n_drifted >= 5:
            return "RETRAIN_RECOMMENDED"
        elif drift_ratio >= 0.05:
            return "MONITOR_CLOSELY"
        else:
            return "NO_ACTION_NEEDED"


class ModelPerformanceMonitor:
    """
    Tracks model prediction accuracy over time.

    Compares predicted directions with actual outcomes
    and generates performance reports.
    """

    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days

    def evaluate_predictions(
        self,
        predictions_df: pd.DataFrame,
        actuals_df: pd.DataFrame,
    ) -> dict:
        """
        Evaluate model predictions against actual outcomes.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Must have: target_date, predicted_direction, confidence
        actuals_df : pd.DataFrame
            Must have: date, close (to compute actual direction)

        Returns
        -------
        dict
            Performance metrics.
        """
        if predictions_df.empty or actuals_df.empty:
            return {"status": "insufficient_data"}

        # Merge predictions with actuals
        merged = predictions_df.merge(
            actuals_df[["date", "close"]],
            left_on="target_date",
            right_on="date",
            how="inner",
        )

        if merged.empty:
            return {"status": "no_matching_dates"}

        # Calculate actual direction
        # (We need the close at prediction_date and target_date)
        correct = (merged["predicted_direction"] == merged["actual_direction"]).sum()
        total = len(merged)

        accuracy = correct / total if total > 0 else 0

        # Confidence calibration
        high_conf = merged[merged["confidence"] > 0.7]
        high_conf_accuracy = 0
        if len(high_conf) > 0:
            high_conf_correct = (
                high_conf["predicted_direction"] == high_conf["actual_direction"]
            ).sum()
            high_conf_accuracy = high_conf_correct / len(high_conf)

        return {
            "status": "evaluated",
            "total_predictions": total,
            "accuracy": round(accuracy, 4),
            "high_confidence_accuracy": round(high_conf_accuracy, 4),
            "high_confidence_count": len(high_conf),
            "period_start": str(merged["target_date"].min()),
            "period_end": str(merged["target_date"].max()),
        }

    def check_model_staleness(
        self, last_trained_at: datetime, max_age_days: int = 30
    ) -> dict:
        """
        Check if model needs retraining based on age.

        Parameters
        ----------
        last_trained_at : datetime
            When the model was last trained.
        max_age_days : int
            Maximum acceptable model age in days.

        Returns
        -------
        dict
            Staleness check result.
        """
        age = (datetime.now() - last_trained_at).days
        is_stale = age > max_age_days

        return {
            "model_age_days": age,
            "max_age_days": max_age_days,
            "is_stale": is_stale,
            "recommendation": "RETRAIN" if is_stale else "OK",
            "last_trained": last_trained_at.isoformat(),
        }
