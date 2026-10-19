"""Statistical utilities and analytical physics formulas for visualization comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_distribution_summary(series: pd.Series) -> dict[str, float]:
    """Compute empirical summary statistics for a 1D numeric series."""
    clean = series.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }

    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "median": float(np.median(clean)),
        "p01": float(np.percentile(clean, 1)),
        "p05": float(np.percentile(clean, 5)),
        "p95": float(np.percentile(clean, 95)),
        "p99": float(np.percentile(clean, 99)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "count": len(clean),
    }


def compute_correlation_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute Pearson correlation matrix across available features."""
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        return pd.DataFrame()
    subset = df[valid_features].apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < 2:
        return pd.DataFrame(index=valid_features, columns=valid_features, data=np.nan)
    return subset.corr(method="pearson")


def compute_normal_vs_attack_stats(
    df: pd.DataFrame,
    features: list[str],
    label_col: str = "ground_truth_attack_active",
) -> pd.DataFrame:
    """Compute mean and std for normal vs attack windows across features."""
    if label_col not in df.columns:
        return pd.DataFrame()

    is_attack = df[label_col].astype(bool)
    normal_df = df[~is_attack]
    attack_df = df[is_attack]

    rows = []
    for f in features:
        if f not in df.columns:
            continue
        n_vals = pd.to_numeric(normal_df[f], errors="coerce").dropna()
        a_vals = pd.to_numeric(attack_df[f], errors="coerce").dropna()

        n_mean = float(n_vals.mean()) if len(n_vals) > 0 else float("nan")
        n_std = float(n_vals.std()) if len(n_vals) > 1 else 0.0
        a_mean = float(a_vals.mean()) if len(a_vals) > 0 else float("nan")
        a_std = float(a_vals.std()) if len(a_vals) > 1 else 0.0

        diff = a_mean - n_mean if not (np.isnan(a_mean) or np.isnan(n_mean)) else float("nan")
        ratio = a_mean / (n_mean + 1e-12) if not (np.isnan(a_mean) or np.isnan(n_mean)) else float("nan")

        rows.append({
            "feature": f,
            "normal_mean": n_mean,
            "normal_std": n_std,
            "attack_mean": a_mean,
            "attack_std": a_std,
            "delta": diff,
            "ratio": ratio,
        })

    return pd.DataFrame(rows)


def evaluate_theoretical_transmission(
    distance_km: np.ndarray,
    alpha_db_km: float = 0.20,
    l_sys_db: float = 1.0,
) -> np.ndarray:
    """Exact channel loss formula: T(L) = 10^(-(alpha*L + L_sys)/10)."""
    total_loss = alpha_db_km * distance_km + l_sys_db
    return np.power(10.0, -total_loss / 10.0)


def evaluate_poisson_pmf(mu: float, max_n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Poisson PMF P(n) = exp(-mu) * mu^n / n! for n in [0, max_n]."""
    n_values = np.arange(0, max_n + 1)
    pmf = stats.poisson.pmf(n_values, mu)
    return n_values, pmf


def evaluate_dead_time_saturation(
    incident_rates: np.ndarray,
    tau_dead_s: float = 50e-9,
) -> np.ndarray:
    """Nonparalyzable dead time count rate formula: r_obs = lambda / (1 + lambda * tau)."""
    return incident_rates / (1.0 + incident_rates * tau_dead_s)
