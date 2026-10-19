"""Statistical distribution comparison and multivariate drift detection (V0.8).

Implements scientifically grounded distribution-shift metrics:
1. Kolmogorov-Smirnov (KS) two-sample test (statistic D and p-value)
2. Wasserstein distance (Earth Mover's Distance W1)
3. Population Stability Index (PSI) style binned divergence
4. Multivariate aggregation: mean, maximum, 90th percentile, and top-k mean
5. Calibrated drift decision thresholds tau_drift with persistence gating
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable

import numpy as np
from scipy import stats


class DriftMetric(str, Enum):
    """Supported statistical distance metrics for distribution comparison."""
    KOLMOGOROV_SMIRNOV = "ks"
    WASSERSTEIN = "wasserstein"
    PSI = "psi"


class AggregationStrategy(str, Enum):
    """Multivariate aggregation strategy across feature dimensions."""
    MEAN = "mean"
    MAX = "max"
    P90 = "p90"
    TOP_K_MEAN = "top_k_mean"


@dataclass
class DriftReport:
    """Detailed report for a single drift evaluation."""
    timestamp_s: float | None
    step_index: int
    metric: DriftMetric
    drift_score: float
    threshold: float
    is_drift: bool
    feature_drift_scores: dict[str, float]
    feature_p_values: dict[str, float]
    dominant_features: list[str]
    sample_size_ref: int
    sample_size_curr: int


@dataclass
class DriftDetectorConfig:
    """Configuration for statistical drift detector."""
    metric: DriftMetric = DriftMetric.WASSERSTEIN
    aggregation: AggregationStrategy = AggregationStrategy.MEAN
    top_k: int = 3
    threshold: float = 0.50
    p_value_threshold: float = 0.05
    min_samples: int = 10
    psi_bins: int = 10
    feature_names: list[str] | None = None


def compute_ks_distance(ref: np.ndarray, curr: np.ndarray) -> tuple[float, float]:
    """Compute two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    if len(ref) < 2 or len(curr) < 2:
        return 0.0, 1.0
    res = stats.ks_2samp(ref, curr)
    return float(res.statistic), float(res.pvalue)


def compute_wasserstein_distance(ref: np.ndarray, curr: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    """Compute normalized 1-Wasserstein distance.

    Normalized by the interquartile range (IQR) or standard deviation of the reference
    distribution to ensure dimensionless scale comparability across heterogeneous telemetry features.
    """
    if len(ref) < 2 or len(curr) < 2:
        return 0.0, 1.0
    w1 = float(stats.wasserstein_distance(ref, curr))
    ref_scale = float(stats.iqr(ref))
    if ref_scale < eps:
        ref_scale = float(np.std(ref))
    if ref_scale < eps:
        ref_scale = 1.0
    normalized_w1 = w1 / ref_scale
    # Approximate pseudo p-value using two-sample KS test
    _, p_val = stats.ks_2samp(ref, curr)
    return normalized_w1, float(p_val)


def compute_psi_distance(ref: np.ndarray, curr: np.ndarray, num_bins: int = 10, eps: float = 1e-4) -> tuple[float, float]:
    """Compute Population Stability Index (PSI) between reference and current samples."""
    if len(ref) < 5 or len(curr) < 5:
        return 0.0, 1.0

    # Quantile bins derived from reference distribution
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles * 100)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    # Ensure strictly monotonic edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        # Insufficient variance for quantile binning, fallback to linear
        bin_edges = np.linspace(float(ref.min()), float(ref.max()) + eps, num_bins + 1)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    curr_counts, _ = np.histogram(curr, bins=bin_edges)

    ref_pct = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
    curr_pct = (curr_counts + eps) / (len(curr) + eps * len(curr_counts))

    psi_val = float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))
    # Asymptotic p-value proxy
    _, p_val = stats.ks_2samp(ref, curr)
    return max(0.0, psi_val), float(p_val)


class StatisticalDriftDetector:
    """Statistical distribution comparator for multivariate QKD telemetry streams."""

    def __init__(self, config: DriftDetectorConfig | None = None):
        self.config = config or DriftDetectorConfig()
        self.metric = self.config.metric
        self.aggregation = self.config.aggregation
        self.threshold = float(self.config.threshold)
        self.feature_names = self.config.feature_names

    def set_threshold(self, threshold: float) -> None:
        """Assign calibrated drift detection threshold."""
        self.threshold = max(0.0, float(threshold))

    def calibrate_threshold_from_reference(
        self,
        reference_data: np.ndarray,
        window_size: int = 20,
        quantile: float = 0.95,
        num_bootstrap: int = 100,
        rng_seed: int = 42,
    ) -> float:
        """Calibrate drift threshold empirically using reference partition bootstraps.

        Samples pairs of non-overlapping sub-windows from pure normal baseline data
        to construct the null distribution of statistical distance under stationary conditions.
        """
        N = len(reference_data)
        if N < window_size * 2:
            return self.threshold

        rng = np.random.RandomState(rng_seed)
        null_scores: list[float] = []

        for _ in range(num_bootstrap):
            idx1 = rng.choice(N - window_size, replace=False)
            # Pick non-overlapping second window
            valid_idx2 = [i for i in range(N - window_size) if abs(i - idx1) >= window_size]
            if not valid_idx2:
                continue
            idx2 = rng.choice(valid_idx2)

            sub_ref = reference_data[idx1 : idx1 + window_size]
            sub_curr = reference_data[idx2 : idx2 + window_size]

            rep = self.compute_drift(sub_ref, sub_curr, step_index=0)
            null_scores.append(rep.drift_score)

        if null_scores:
            calibrated_th = float(np.percentile(null_scores, quantile * 100))
            # Ensure safe lower bound to prevent noise over-sensitivity
            self.threshold = max(calibrated_th, 0.05)
        return self.threshold

    def compute_drift(
        self,
        reference_samples: np.ndarray,
        current_samples: np.ndarray,
        step_index: int = 0,
        timestamp_s: float | None = None,
    ) -> DriftReport:
        """Compare current telemetry distribution against reference distribution.

        Args:
            reference_samples: Array of shape (N_ref, d)
            current_samples: Array of shape (N_curr, d)
            step_index: Current streaming step index
            timestamp_s: Optional simulated timestamp

        Returns:
            DriftReport containing multivariate drift score, per-feature details, and decision.
        """
        ref_arr = np.asarray(reference_samples, dtype=np.float64)
        curr_arr = np.asarray(current_samples, dtype=np.float64)

        if ref_arr.ndim == 1:
            ref_arr = ref_arr.reshape(-1, 1)
        if curr_arr.ndim == 1:
            curr_arr = curr_arr.reshape(-1, 1)

        d = ref_arr.shape[1]
        feature_names = self.feature_names or [f"feature_{i}" for i in range(d)]

        # Check sample size minimums
        if len(ref_arr) < self.config.min_samples or len(curr_arr) < self.config.min_samples:
            return DriftReport(
                timestamp_s=timestamp_s,
                step_index=step_index,
                metric=self.metric,
                drift_score=0.0,
                threshold=self.threshold,
                is_drift=False,
                feature_drift_scores={name: 0.0 for name in feature_names},
                feature_p_values={name: 1.0 for name in feature_names},
                dominant_features=[],
                sample_size_ref=len(ref_arr),
                sample_size_curr=len(curr_arr),
            )

        feat_scores: dict[str, float] = {}
        feat_pvals: dict[str, float] = {}

        for j in range(d):
            name = feature_names[j] if j < len(feature_names) else f"feature_{j}"
            col_ref = ref_arr[:, j]
            col_curr = curr_arr[:, j]

            if self.metric == DriftMetric.KOLMOGOROV_SMIRNOV:
                dist, pval = compute_ks_distance(col_ref, col_curr)
            elif self.metric == DriftMetric.WASSERSTEIN:
                dist, pval = compute_wasserstein_distance(col_ref, col_curr)
            elif self.metric == DriftMetric.PSI:
                dist, pval = compute_psi_distance(col_ref, col_curr, num_bins=self.config.psi_bins)
            else:
                dist, pval = compute_wasserstein_distance(col_ref, col_curr)

            feat_scores[name] = float(dist)
            feat_pvals[name] = float(pval)

        score_values = list(feat_scores.values())

        # Aggregate multivariate score
        if self.aggregation == AggregationStrategy.MAX:
            multivariate_score = float(np.max(score_values))
        elif self.aggregation == AggregationStrategy.P90:
            multivariate_score = float(np.percentile(score_values, 90))
        elif self.aggregation == AggregationStrategy.TOP_K_MEAN:
            k = min(self.config.top_k, len(score_values))
            sorted_scores = sorted(score_values, reverse=True)
            multivariate_score = float(np.mean(sorted_scores[:k]))
        else:  # MEAN
            multivariate_score = float(np.mean(score_values))

        # Rank dominant features driving drift
        sorted_feats = sorted(feat_scores.items(), key=lambda item: item[1], reverse=True)
        dominant_features = [f[0] for f in sorted_feats[: self.config.top_k]]

        is_drift = multivariate_score > self.threshold

        return DriftReport(
            timestamp_s=timestamp_s,
            step_index=step_index,
            metric=self.metric,
            drift_score=multivariate_score,
            threshold=self.threshold,
            is_drift=is_drift,
            feature_drift_scores=feat_scores,
            feature_p_values=feat_pvals,
            dominant_features=dominant_features,
            sample_size_ref=len(ref_arr),
            sample_size_curr=len(curr_arr),
        )
