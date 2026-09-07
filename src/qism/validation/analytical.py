"""Analytical relations used to validate the simulator.

These are the closed-form expressions the Monte Carlo output must reproduce:

    channel transmittance:  eta_ch = 10^(-(alpha*L + L_sys)/10)
    misalignment:           p_mis = sin^2(theta)
    coherent-state gain:    Q(mu) = 1 - exp(-mu * eta_ch * eta_d)
                            (per detector; combined with dark counts below)
    vacuum gain:            Q_vac = 1 - (1 - p_dark0)(1 - p_dark1)
    decoy reconstruction:   Q_mu_pred = sum_n P(n|mu) * Y_n   (truncated)
    timing width:           sigma_t (configured jitter)
    sifted QBER:            ~ p_mis for the dominant (misalignment) term
"""

from __future__ import annotations

import numpy as np


def channel_transmittance(alpha_db_per_km: float, distance_km: float, l_sys_db: float) -> float:
    """eta_ch = 10^(-(alpha*L + L_sys)/10)."""
    return float(10.0 ** (-(alpha_db_per_km * distance_km + l_sys_db) / 10.0))


def misalignment_probability(theta_deg: float) -> float:
    """p_mis = sin^2(theta), theta in degrees."""
    return float(np.sin(np.deg2rad(theta_deg)) ** 2)


def gain_from_coherent_state(mu: float, eta_ch: float, eta_d: float) -> float:
    """Single-detector signal click probability: 1 - exp(-mu * eta_ch * eta_d)."""
    return float(-np.expm1(-mu * eta_ch * eta_d))


def gain_total(
    mu: float,
    eta_ch: float,
    eta_d: float,
    p_dark0: float,
    p_dark1: float,
) -> float:
    """Combined gain for one routed signal detector plus both dark processes.

    The signal photon reaches exactly one detector with efficiency eta_d;
    dark counts on both detectors are independent:
        Q = 1 - (1 - P_sig)(1 - p_dark0)(1 - p_dark1)
    """
    p_sig = gain_from_coherent_state(mu, eta_ch, eta_d)
    return 1.0 - (1.0 - p_sig) * (1.0 - p_dark0) * (1.0 - p_dark1)


def gain_vacuum(p_dark0: float, p_dark1: float) -> float:
    """Vacuum gain from dark counts only: 1 - (1-p_dark0)(1-p_dark1)."""
    return 1.0 - (1.0 - p_dark0) * (1.0 - p_dark1)


def q_mu_pred(yields: dict[int, float], mu: float, max_photon_number: int) -> float:
    """Reconstructed gain Q_mu_pred = sum_{n=0}^{n_max} P(n|mu) * Y_n.

    Yield entries with no samples (NaN) are skipped; the reconstruction is
    exact for sufficiently sampled photon numbers.
    """
    pred = 0.0
    for n in range(0, max_photon_number + 1):
        y = yields.get(n, np.nan)
        if np.isnan(y):
            continue
        pred += _poisson_pmf(n, mu) * float(y)
    return float(pred)


def _poisson_pmf(n: int, mu: float) -> float:
    import math

    return math.exp(-mu) * mu**n / math.factorial(n)


def poisson_probs(mu: float, n_max: int) -> np.ndarray:
    """P(n|mu) for n = 0..n_max."""
    return np.array([_poisson_pmf(n, mu) for n in range(n_max + 1)], dtype=np.float64)


def decoy_residual(q_observed: float, q_pred: float) -> float:
    """R = Q_obs - Q_pred."""
    return float(q_observed - q_pred)


# --------------------------------------------------------------------------- #
# V0.2 detector-dynamics analytical relations (dead-time theory)
# --------------------------------------------------------------------------- #


def observed_rate_nonparalyzable(lambda_hz: float, tau_s: float) -> float:
    """Ideal observed count rate of a nonparalyzable detector fed by a
    homogeneous Poisson process: r_obs = lambda / (1 + lambda * tau).

    Valid when dark counts, afterpulsing, gating and pulse discretization are
    disabled (see the validation experiments).
    """
    lambda_hz = float(lambda_hz)
    tau_s = float(tau_s)
    if lambda_hz <= 0.0:
        return 0.0
    return lambda_hz / (1.0 + lambda_hz * tau_s)


def interarrival_mean_nonparalyzable(lambda_hz: float, tau_s: float) -> float:
    """E[Delta_observed] = tau + 1/lambda for the ideal nonparalyzable model
    (accepted clicks are spaced by tau plus an exponential remainder)."""
    return float(tau_s) + 1.0 / float(lambda_hz)


def interarrival_std_nonparalyzable(lambda_hz: float) -> float:
    """Var(Delta_observed) = 1/lambda^2 -> std = 1/lambda."""
    return 1.0 / float(lambda_hz)


def busy_fraction_nonparalyzable(n_clicks: int, tau_s: float, t_obs_s: float) -> float:
    """B ~= N_clicks * tau / T_obs (nonparalyzable: disjoint intervals)."""
    if t_obs_s <= 0.0:
        return float("nan")
    return float(n_clicks) * float(tau_s) / float(t_obs_s)


def tau_eff_rate_dependent(
    lambda_hz: float, tau_0_s: float, delta_tau_s: float, lambda_c_hz: float
) -> float:
    """tau_eff(lambda) = tau_0 + delta_tau * lambda / (lambda + lambda_c)."""
    lam = max(0.0, float(lambda_hz))
    return float(tau_0_s) + float(delta_tau_s) * lam / (lam + float(lambda_c_hz))
