"""Physics-informed window feature extraction.

Window features are computed from event-level records only — never fabricated
independently. Every feature is traceable back to underlying events.

Conventions (documented):
  * detector_i_count    — events where detector i clicked (double clicks count
                          for both detectors).
  * N_detection         — events with at least one click = C0 + C1 - N_double.
  * double_click_rate   — N_double / N_detection.
  * detector imbalance  — Delta_c = (C0 - C1) / (C0 + C1); NaN when C0+C1=0.
  * QBER                — N_errors / N_sifted over sifted single-click events
                          (double clicks excluded per the V0.1 policy);
                          NaN when N_sifted = 0.
  * timing moments      — population (bias=True) moments; kurtosis is the
                          excess (Fisher) kurtosis, matching
                          scipy.stats.kurtosis(..., fisher=True, bias=True).
  * dark_count_rate     — empirical per-detector dark-click probability per
                          pulse (dimensionless; the configured rate is [Hz]).
  * decoy residuals     — R_I = Q_obs,I - Q_pred,I with
                          Q_pred,I = sum_n P(n|mu_I) * Y_n (truncated at
                          max_photon_number).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

INTENSITY_CLASSES = ("signal", "decoy", "vacuum")
MU_BY_CLASS = ("mu_signal", "mu_decoy", "mu_vacuum")


def _safe_ratio(numer: float, denom: float) -> float:
    """numer / denom; NaN (never 0) when the denominator is zero."""
    if denom == 0:
        return float("nan")
    return float(numer) / float(denom)


def _timing_moments(offsets: np.ndarray) -> dict[str, float]:
    """Population moments of timing offsets; excess (Fisher) kurtosis."""
    if offsets.size == 0:
        return {"mean": np.nan, "std": np.nan, "skewness": np.nan, "kurtosis": np.nan}
    mean = float(np.mean(offsets))
    std = float(np.std(offsets, ddof=0))
    if offsets.size < 3:
        return {"mean": mean, "std": std, "skewness": np.nan, "kurtosis": np.nan}
    sk = float(skew(offsets, bias=True))
    ku = float(kurtosis(offsets, fisher=True, bias=True))
    return {"mean": mean, "std": std, "skewness": sk, "kurtosis": ku}


def window_features(
    events: pd.DataFrame,
    window_size: int,
    max_photon_number: int = 10,
    timing_histogram_bins: int = 20,
    mu_signal: float = 0.5,
    mu_decoy: float = 0.1,
    mu_vacuum: float = 0.0,
    repetition_rate_hz: float = 1_000_000.0,
    interarrival_histogram_bins: int = 30,
    interarrival_histogram_scale: str = "linear",
    fano_sub_bins: int = 64,
) -> dict[str, Any]:
    """Compute the window feature vector from one window's events.

    V0.2 detector-dynamics features follow these documented conventions:
      * observation time      — T_obs = n_pulses / f_rep (the window's span).
      * candidate_event_rate  — per-detector candidates offered / T_obs.
      * dead_time_loss_rate   — dead-rejected candidates / candidates; NaN
        when there were no candidates (never silently 0).
      * busy fraction         — sum of per-avalanche effective dead durations
        inside the window divided by T_obs. Nonparalyzable dead intervals do
        not overlap, so this equals the interval-union busy time; the exact
        union method (valid even for overlapping-capable future models) is
        implemented on Detector.busy_time() and used by the validation probe.
      * ready fraction        — 1 - busy fraction (≈, boundary effects only).
      * inter-arrival times   — gaps between consecutive accepted avalanches
        of the same detector, computed from the per-detector arrival times
        inside the window; percentiles via numpy.percentile.
      * effective_dead_time   — mean tau_eff actually applied to the window's
        avalanches (from the per-click dead durations).
      * afterpulse_rate_ground_truth — detected afterpulses / primary
        avalanches. SIMULATOR GROUND TRUTH: never use as an ML input.
      * fano factor           — Var(count)/mean(count) of the combined click
        count per fixed sub-bin of pulses (Poisson baseline: 1).
      * autocorrelation       — Pearson autocorrelation of the per-pulse
        click indicator series for lags 1..5 (lag 0 baseline: 1).

    Fields documented as ground truth (afterpulse_rate_ground_truth, ...)
    must be excluded from any future ML feature matrix.
    """
    n_pulses = len(events)
    alice_basis = events["alice_basis"].to_numpy()
    bob_basis = events["bob_basis"].to_numpy()
    intensity = events["intensity_class"].to_numpy()
    photon_number = events["photon_number"].to_numpy()
    click0 = events["detector_0_click"].to_numpy()
    click1 = events["detector_1_click"].to_numpy()
    clicked = click0 | click1
    double = click0 & click1
    sifted = alice_basis == bob_basis

    # ---------------- counts ---------------- #
    n_signal = int(np.sum(intensity == "signal"))
    n_decoy = int(np.sum(intensity == "decoy"))
    n_vacuum = int(np.sum(intensity == "vacuum"))
    n_d0 = int(np.sum(click0))
    n_d1 = int(np.sum(click1))
    n_double = int(np.sum(double))
    n_detection = int(np.sum(clicked))
    single = clicked & ~double

    # ---------------- sifting / QBER ---------------- #
    # V0.1 policy: double clicks are excluded from the sifted key.
    key_mask = sifted & single
    n_sifted = int(np.sum(key_mask))
    bit_error = events["bit_error"].to_numpy(dtype=np.float64)
    err_mask = key_mask & (bit_error == 1.0)
    n_errors = int(np.sum(err_mask))
    n_sifted_z = int(np.sum(key_mask & (alice_basis == 0)))
    n_errors_z = int(np.sum(err_mask & (alice_basis == 0)))
    n_sifted_x = int(np.sum(key_mask & (alice_basis == 1)))
    n_errors_x = int(np.sum(err_mask & (alice_basis == 1)))

    qber = _safe_ratio(n_errors, n_sifted)
    qber_z = _safe_ratio(n_errors_z, n_sifted_z)
    qber_x = _safe_ratio(n_errors_x, n_sifted_x)

    # ---------------- gains ---------------- #
    gain = {}
    for cls in INTENSITY_CLASSES:
        mask = intensity == cls
        n_sent = int(np.sum(mask))
        n_click = int(np.sum(mask & clicked))
        gain[cls] = _safe_ratio(n_click, n_sent)

    # ---------------- yields (ground truth) ---------------- #
    yields: dict[int, float] = {}
    for n in range(0, max_photon_number + 1):
        mask = photon_number == n
        n_sent = int(np.sum(mask))
        n_click = int(np.sum(mask & clicked))
        yields[n] = _safe_ratio(n_click, n_sent)

    # ---------------- decoy residuals ---------------- #
    residual = {}
    q_pred = {}
    for cls, mu in zip(INTENSITY_CLASSES, (mu_signal, mu_decoy, mu_vacuum)):
        pred = 0.0
        for n in range(0, max_photon_number + 1):
            p_n = float(np.exp(-mu) * mu**n / math.factorial(n))
            pred += p_n * yields[n]
        q_pred[cls] = pred
        residual[cls] = gain[cls] - pred

    # ---------------- detector statistics ---------------- #
    imbalance = _safe_ratio(n_d0 - n_d1, n_d0 + n_d1)
    double_rate = _safe_ratio(n_double, n_detection)

    dark0 = events["detector_0_click"].to_numpy() & _source_has_dark(events, 0)
    dark1 = events["detector_1_click"].to_numpy() & _source_has_dark(events, 1)
    n_dark0 = int(np.sum(dark0))
    n_dark1 = int(np.sum(dark1))
    n_dark_events = int(np.sum(dark0 | dark1))
    dark_fraction = _safe_ratio(n_dark_events, n_pulses)
    dark_per_detector = _safe_ratio(n_dark0 + n_dark1, 2.0 * n_pulses)

    # ---------------- timing ---------------- #
    offsets = events["timing_offset_s"].to_numpy(dtype=np.float64)
    offsets = offsets[~np.isnan(offsets)]
    moments = _timing_moments(offsets)
    if offsets.size == 0:
        hist_counts, hist_edges = [], []
    else:
        hist_counts, hist_edges = np.histogram(offsets, bins=timing_histogram_bins)
        hist_counts = hist_counts.tolist()
        hist_edges = hist_edges.tolist()

    # ---------------- photon statistics (ground truth) ---------------- #
    pn = photon_number.astype(np.float64)
    mean_pn = float(np.mean(pn)) if n_pulses else np.nan
    variance_pn = float(np.var(pn, ddof=0)) if n_pulses else np.nan
    vacuum_fraction = _safe_ratio(int(np.sum(photon_number == 0)), n_pulses)
    single_fraction = _safe_ratio(int(np.sum(photon_number == 1)), n_pulses)
    multi_fraction = _safe_ratio(int(np.sum(photon_number >= 2)), n_pulses)

    # ---------------- channel ---------------- #
    mean_transmittance = float(np.mean(events["channel_transmittance"].to_numpy()))

    # ---------------- V0.2 detector dynamics ---------------- #
    dyn = _detector_dynamics_features(
        events,
        n_pulses=n_pulses,
        repetition_rate_hz=repetition_rate_hz,
        interarrival_histogram_bins=interarrival_histogram_bins,
        interarrival_histogram_scale=interarrival_histogram_scale,
        fano_sub_bins=fano_sub_bins,
    )

    # ---------------- V0.4 infrastructure telemetry ---------------- #
    infra = _infrastructure_features(
        events,
        n_pulses=n_pulses,
        repetition_rate_hz=repetition_rate_hz,
    )

    features: dict[str, Any] = {
        # protocol
        "qber": qber,
        "qber_z": qber_z,
        "qber_x": qber_x,
        "sifted_count": n_sifted,
        "sifted_rate": _safe_ratio(n_sifted, n_pulses),
        "error_count": n_errors,
        # gain
        "signal_gain": gain["signal"],
        "decoy_gain": gain["decoy"],
        "vacuum_gain": gain["vacuum"],
        # decoy residuals
        "signal_decoy_residual": residual["signal"],
        "decoy_decoy_residual": residual["decoy"],
        "vacuum_residual": residual["vacuum"],
        # detector
        "detector_0_count": n_d0,
        "detector_1_count": n_d1,
        "total_count": n_detection,
        "detector_imbalance": imbalance,
        "double_click_rate": double_rate,
        "dark_count_rate": dark_per_detector,
        "dark_click_fraction": dark_fraction,
        # timing
        "mean_arrival_offset": moments["mean"],
        "std_arrival_offset": moments["std"],
        "timing_skewness": moments["skewness"],
        "timing_kurtosis": moments["kurtosis"],
        "timing_histogram_counts": hist_counts,
        "timing_histogram_edges": hist_edges,
        # photon statistics (simulator ground truth)
        "mean_photon_number": mean_pn,
        "photon_variance": variance_pn,
        "vacuum_fraction": vacuum_fraction,
        "single_photon_fraction": single_fraction,
        "multiphoton_fraction": multi_fraction,
        # channel
        "mean_channel_transmittance": mean_transmittance,
        # counts needed to reconstruct every feature
        "n_pulses": n_pulses,
        "n_signal": n_signal,
        "n_decoy": n_decoy,
        "n_vacuum": n_vacuum,
        "n_sifted_z": n_sifted_z,
        "n_sifted_x": n_sifted_x,
        "n_errors_z": n_errors_z,
        "n_errors_x": n_errors_x,
        "n_double": n_double,
        "n_detection": n_detection,
        "n_dark_0": n_dark0,
        "n_dark_1": n_dark1,
        # V0.2 detector dynamics
        **dyn,
        # V0.4 infrastructure & environmental telemetry
        **infra,
    }
    for n in range(0, max_photon_number + 1):
        features[f"yield_{n}"] = yields[n]
    return features


def _detector_dynamics_features(
    events: pd.DataFrame,
    *,
    n_pulses: int,
    repetition_rate_hz: float,
    interarrival_histogram_bins: int,
    interarrival_histogram_scale: str,
    fano_sub_bins: int,
) -> dict[str, Any]:
    """V0.2 detector-dynamics window features (see window_features docstring)."""
    period = 1.0 / repetition_rate_hz
    t_obs = n_pulses * period
    if t_obs <= 0:
        t_obs = float("nan")

    click0 = events["detector_0_click"].to_numpy()
    click1 = events["detector_1_click"].to_numpy()
    clicked = click0 | click1

    cand0 = events["detector_0_candidate_count"].to_numpy(dtype=np.int64)
    cand1 = events["detector_1_candidate_count"].to_numpy(dtype=np.int64)
    rej0 = events["detector_0_dead_rejected_count"].to_numpy(dtype=np.int64)
    rej1 = events["detector_1_dead_rejected_count"].to_numpy(dtype=np.int64)
    n_cand0 = int(cand0.sum())
    n_cand1 = int(cand1.sum())
    n_rej0 = int(rej0.sum())
    n_rej1 = int(rej1.sum())
    n_cand = n_cand0 + n_cand1
    n_rej = n_rej0 + n_rej1
    n_accepted = int(clicked.sum())  # accepted avalanches (<= 1 per pulse per detector)

    def rate(count: int) -> float:
        return _safe_ratio(float(count), t_obs)

    # per-detector busy fraction from per-avalanche dead durations
    dd0 = events["detector_0_dead_duration_s"].to_numpy(dtype=np.float64)
    dd1 = events["detector_1_dead_duration_s"].to_numpy(dtype=np.float64)
    busy0 = float(np.nansum(dd0)) if n_pulses else 0.0
    busy1 = float(np.nansum(dd1)) if n_pulses else 0.0
    busy_frac0 = _safe_ratio(busy0, t_obs)
    busy_frac1 = _safe_ratio(busy1, t_obs)

    # per-detector inter-arrival statistics (consecutive accepted avalanches)
    def interarrival(times_col: str) -> dict[str, Any]:
        arr = events[times_col].to_numpy(dtype=np.float64)
        arr = np.sort(arr[~np.isnan(arr)])
        gaps = np.diff(arr)
        out: dict[str, Any] = {}
        if gaps.size == 0:
            base = {
                "mean": np.nan,
                "std": np.nan,
                "p01": np.nan,
                "p05": np.nan,
                "p25": np.nan,
                "p50": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "p99": np.nan,
            }
        else:
            base = {
                "mean": float(np.mean(gaps)),
                "std": float(np.std(gaps, ddof=1)) if gaps.size > 1 else float(np.nan),
                "p01": float(np.percentile(gaps, 1)),
                "p05": float(np.percentile(gaps, 5)),
                "p25": float(np.percentile(gaps, 25)),
                "p50": float(np.percentile(gaps, 50)),
                "p75": float(np.percentile(gaps, 75)),
                "p95": float(np.percentile(gaps, 95)),
                "p99": float(np.percentile(gaps, 99)),
            }
        out.update(base)
        # histogram of inter-arrival gaps (from the actual click times)
        if gaps.size == 0:
            out["hist_counts"] = []
            out["hist_edges"] = []
        else:
            if interarrival_histogram_scale == "log" and gaps.min() > 0:
                edges = np.logspace(
                    np.log10(gaps.min()), np.log10(gaps.max()), interarrival_histogram_bins + 1
                )
                counts, edges = np.histogram(gaps, bins=edges)
            else:
                counts, edges = np.histogram(gaps, bins=interarrival_histogram_bins)
            out["hist_counts"] = counts.tolist()
            out["hist_edges"] = edges.tolist()
        return out

    ia0 = interarrival("detector_0_arrival_time_s")
    ia1 = interarrival("detector_1_arrival_time_s")

    # effective dead time: mean tau_eff applied to the window's avalanches
    eff_dd = np.concatenate(
        [dd0[click0], dd1[click1 & ~click0]]
    )  # per-avalanche (double: D0's value only, as in the legacy convention)
    effective_dead_time = float(np.mean(eff_dd)) if eff_dd.size else float("nan")

    # Fano factor over the per-pulse click indicator (combined detectors)
    ind = clicked.astype(np.float64)
    bin_size = max(1, int(n_pulses) // max(1, fano_sub_bins))
    n_bins = max(1, int(n_pulses) // bin_size)
    counts = np.array([ind[i * bin_size : (i + 1) * bin_size].sum() for i in range(n_bins)])
    fano = _safe_ratio(float(np.var(counts, ddof=0)), float(np.mean(counts)))

    # autocorrelation of the click indicator series, lags 1..5
    acf: dict[str, float] = {}
    mu = float(np.mean(ind)) if n_pulses else 0.0
    var = float(np.var(ind, ddof=0)) if n_pulses else 0.0
    if var > 0 and n_pulses > 1:
        xc = ind - mu
        for lag in range(1, 6):
            num = float(np.mean(xc[:-lag] * xc[lag:])) if lag < len(xc) else 0.0
            acf[f"autocorr_lag{lag}"] = num / var
    else:
        for lag in range(1, 6):
            acf[f"autocorr_lag{lag}"] = float("nan")

    # afterpulse ground-truth aggregates
    ap_detected = events["afterpulse_detected"].to_numpy(dtype=bool)
    n_ap_detected = int(ap_detected.sum())
    ct = events["click_type"].to_numpy()
    has_ap_label = np.char.find(ct.astype(str), "afterpulse") >= 0
    n_primary = int(np.sum(clicked & ~has_ap_label))
    ap_rate_gt = _safe_ratio(n_ap_detected, n_primary)
    delays = events["afterpulse_delay_s"].to_numpy(dtype=np.float64)
    delays = delays[ap_detected & ~np.isnan(delays)]
    mean_ap_delay = float(np.mean(delays)) if delays.size else float("nan")

    return {
        # detector dynamics / loss rates
        "dead_time_loss_rate": _safe_ratio(float(n_rej), float(n_cand)),
        "candidate_event_rate": rate(n_cand),
        "accepted_event_rate": rate(n_accepted),
        "observed_click_rate": rate(n_accepted),
        "count_rate": rate(n_accepted),
        "effective_dead_time": effective_dead_time,
        # utilization (fractions of T_obs)
        "detector_0_busy_fraction": busy_frac0,
        "detector_1_busy_fraction": busy_frac1,
        "detector_0_ready_fraction": 1.0 - busy_frac0 if not np.isnan(busy_frac0) else np.nan,
        "detector_1_ready_fraction": 1.0 - busy_frac1 if not np.isnan(busy_frac1) else np.nan,
        "detector_0_dead_fraction": busy_frac0,
        "detector_1_dead_fraction": busy_frac1,
        # inter-arrival statistics
        "detector_0_mean_interarrival": ia0["mean"],
        "detector_0_std_interarrival": ia0["std"],
        "detector_1_mean_interarrival": ia1["mean"],
        "detector_1_std_interarrival": ia1["std"],
        "detector_0_p01_interarrival": ia0["p01"],
        "detector_0_p05_interarrival": ia0["p05"],
        "detector_0_p50_interarrival": ia0["p50"],
        "detector_0_p95_interarrival": ia0["p95"],
        "detector_0_p99_interarrival": ia0["p99"],
        "detector_1_p01_interarrival": ia1["p01"],
        "detector_1_p05_interarrival": ia1["p05"],
        "detector_1_p50_interarrival": ia1["p50"],
        "detector_1_p95_interarrival": ia1["p95"],
        "detector_1_p99_interarrival": ia1["p99"],
        # histograms of inter-arrival gaps
        "detector_0_interarrival_histogram_counts": ia0["hist_counts"],
        "detector_0_interarrival_histogram_edges": ia0["hist_edges"],
        "detector_1_interarrival_histogram_counts": ia1["hist_counts"],
        "detector_1_interarrival_histogram_edges": ia1["hist_edges"],
        # click-process statistics
        "fano_factor": fano,
        **acf,
        # afterpulse (simulator ground truth — excluded from ML inputs)
        "afterpulse_rate_ground_truth": ap_rate_gt,
        "mean_afterpulse_delay": mean_ap_delay,
        # reconstruction counts
        "n_candidate_0": n_cand0,
        "n_candidate_1": n_cand1,
        "n_dead_rejected_0": n_rej0,
        "n_dead_rejected_1": n_rej1,
        "n_afterpulse_detected": n_ap_detected,
        "n_primary_avalanches": n_primary,
    }


def _infrastructure_features(
    events: pd.DataFrame,
    *,
    n_pulses: int,
    repetition_rate_hz: float,
) -> dict[str, Any]:
    """V0.4 infrastructure & environmental monitoring telemetry."""
    period = 1.0 / repetition_rate_hz
    t_obs = n_pulses * period if n_pulses > 0 else float("nan")

    # Observable Sensor Summaries
    def col_stats(col: str) -> tuple[float, float, float]:
        if col not in events.columns:
            return float("nan"), float("nan"), float("nan")
        vals = events[col].to_numpy(dtype=np.float64)
        valid = vals[~np.isnan(vals)]
        if valid.size == 0:
            return float("nan"), float("nan"), float("nan")
        mean_v = float(np.mean(valid))
        std_v = float(np.std(valid)) if valid.size > 1 else 0.0
        drift_v = float(valid[-1] - valid[0]) if valid.size > 1 else 0.0
        return mean_v, std_v, drift_v

    t_mean, t_std, t_drift = col_stats("sensor_temperature_c")
    h_mean, h_std, _ = col_stats("sensor_humidity_pct")
    v_mean, v_std, _ = col_stats("sensor_vibration_g")
    bg_mean, bg_std, bg_drift = col_stats("sensor_background_light_lux")
    clk_mean, clk_std, clk_drift = col_stats("clock_offset_s")

    # Channel metrics
    trans = events["channel_transmittance"].to_numpy(dtype=np.float64)
    ch_mean = float(np.mean(trans)) if trans.size else float("nan")
    ch_var = float(np.var(trans)) if trans.size else float("nan")
    ch_drift = float(trans[-1] - trans[0]) if trans.size > 1 else 0.0
    est_loss_db = float(-10.0 * math.log10(max(1e-12, ch_mean))) if not np.isnan(ch_mean) and ch_mean > 0 else float("nan")

    # Source metrics
    mu_arr = events["mu"].to_numpy(dtype=np.float64)
    src_mean = float(np.mean(mu_arr)) if mu_arr.size else float("nan")
    src_var = float(np.var(mu_arr)) if mu_arr.size else float("nan")
    src_stability = float(src_var / max(1e-6, src_mean**2)) if src_mean > 0 else 0.0

    # Rates
    click0 = events["detector_0_click"].to_numpy()
    click1 = events["detector_1_click"].to_numpy()
    n_det = int(np.sum(click0 | click1))
    detection_rate_hz = float(n_det / t_obs) if t_obs > 0 else float("nan")
    sifted_count = int(np.sum((events["alice_basis"] == events["bob_basis"]) & (click0 ^ click1)))
    sifted_key_rate_hz = float(sifted_count / t_obs) if t_obs > 0 else float("nan")

    return {
        "sensor_temperature_mean": t_mean,
        "sensor_temperature_std": t_std,
        "sensor_temperature_drift": t_drift,
        "sensor_humidity_mean": h_mean,
        "sensor_humidity_std": h_std,
        "sensor_vibration_mean": v_mean,
        "sensor_vibration_std": v_std,
        "sensor_background_light_mean": bg_mean,
        "sensor_background_light_std": bg_std,
        "sensor_background_light_drift": bg_drift,
        "clock_offset_mean": clk_mean,
        "clock_offset_std": clk_std,
        "clock_drift_rate": float(clk_drift / t_obs) if t_obs > 0 else 0.0,
        "estimated_channel_loss_db": est_loss_db,
        "channel_transmittance_variance": ch_var,
        "channel_transmittance_drift": ch_drift,
        "source_intensity_mean": src_mean,
        "source_intensity_variance": src_var,
        "source_stability": src_stability,
        "detection_rate_hz": detection_rate_hz,
        "sifted_key_rate_hz": sifted_key_rate_hz,
    }


def _source_has_dark(events: pd.DataFrame, detector_index: int) -> np.ndarray:
    """Mask of events whose clicked detector includes a dark-count contribution."""
    ct = events["click_type"].to_numpy()
    if detector_index == 0:
        return np.isin(ct, ["d0_dark", "d0_signal_dark", "double_dark", "double_mixed"])
    return np.isin(ct, ["d1_dark", "d1_signal_dark", "double_dark", "double_mixed"])


GROUND_TRUTH_WINDOW_PREFIXES = ("ground_truth_", "yield_", "true_")
GROUND_TRUTH_WINDOW_COLUMNS = frozenset(
    [
        "afterpulse_rate_ground_truth",
        "mean_afterpulse_delay",
        "mean_photon_number",
        "photon_variance",
        "vacuum_fraction",
        "single_photon_fraction",
        "multiphoton_fraction",
        "mean_channel_transmittance",
        "ground_truth_attack_active",
        "ground_truth_attack_type",
        "ground_truth_attack_phase",
        "ground_truth_attack_fraction",
        "ground_truth_mean_severity",
        "ground_truth_attack_id",
    ]
    + [f"yield_{n}" for n in range(11)]
)

IDENTIFIER_WINDOW_COLUMNS = frozenset(
    [
        "run_id",
        "scenario_id",
        "window_id",
        "window_start_pulse",
        "window_end_pulse",
    ]
)


def get_observable_feature_names(columns: list[str] | None = None) -> list[str]:
    """Return strictly observable telemetry feature column names.

    Excludes all simulator ground truth, attack labels, and identifiers.
    ML anomaly detectors and classifiers must only train and evaluate on
    these observable features.
    """
    if columns is None:
        # Generate full canonical observable feature list
        from qism.config import QKDConfig
        from qism.telemetry.windows import WindowAggregator
        # Return standard observable keys
        sample_keys = [
            "qber", "qber_z", "qber_x", "sifted_count", "sifted_rate", "error_count",
            "signal_gain", "decoy_gain", "vacuum_gain",
            "signal_decoy_residual", "decoy_decoy_residual", "vacuum_residual",
            "detector_0_count", "detector_1_count", "total_count", "detector_imbalance",
            "double_click_rate", "dark_count_rate", "dark_click_fraction",
            "mean_arrival_offset", "std_arrival_offset", "timing_skewness", "timing_kurtosis",
            "dead_time_loss_rate", "candidate_event_rate", "accepted_event_rate",
            "observed_click_rate", "count_rate", "effective_dead_time",
            "detector_0_busy_fraction", "detector_1_busy_fraction",
            "detector_0_ready_fraction", "detector_1_ready_fraction",
            "detector_0_dead_fraction", "detector_1_dead_fraction",
            "detector_0_mean_interarrival", "detector_0_std_interarrival",
            "detector_1_mean_interarrival", "detector_1_std_interarrival",
            "detector_0_p01_interarrival", "detector_0_p05_interarrival",
            "detector_0_p50_interarrival", "detector_0_p95_interarrival", "detector_0_p99_interarrival",
            "detector_1_p01_interarrival", "detector_1_p05_interarrival",
            "detector_1_p50_interarrival", "detector_1_p95_interarrival", "detector_1_p99_interarrival",
            "fano_factor", "autocorr_lag1", "autocorr_lag2", "autocorr_lag3", "autocorr_lag4", "autocorr_lag5",
            # V0.4 infrastructure & sensor observables
            "sensor_temperature_mean", "sensor_temperature_std", "sensor_temperature_drift",
            "sensor_humidity_mean", "sensor_humidity_std",
            "sensor_vibration_mean", "sensor_vibration_std",
            "sensor_background_light_mean", "sensor_background_light_std", "sensor_background_light_drift",
            "clock_offset_mean", "clock_offset_std", "clock_drift_rate",
            "estimated_channel_loss_db", "channel_transmittance_variance", "channel_transmittance_drift",
            "source_intensity_mean", "source_intensity_variance", "source_stability",
            "detection_rate_hz", "sifted_key_rate_hz",
        ]
        return sample_keys

    return [
        col for col in columns
        if col not in IDENTIFIER_WINDOW_COLUMNS
        and col not in GROUND_TRUTH_WINDOW_COLUMNS
        and not any(col.startswith(p) for p in GROUND_TRUTH_WINDOW_PREFIXES)
        and not col.startswith("n_")
        and not col.endswith("_histogram_counts")
        and not col.endswith("_histogram_edges")
    ]


def get_ground_truth_feature_names(columns: list[str] | None = None) -> list[str]:
    """Return ground truth and label column names from a window DataFrame."""
    if columns is None:
        return sorted(list(GROUND_TRUTH_WINDOW_COLUMNS))
    return [
        col for col in columns
        if col in GROUND_TRUTH_WINDOW_COLUMNS
        or any(col.startswith(p) for p in GROUND_TRUTH_WINDOW_PREFIXES)
    ]

