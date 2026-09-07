"""The decoy-state BB84 stochastic event simulator (V0.3).

Two simulation engines share one configuration and event schema:

* **V0.1 vectorized path** — used when *no* detector dynamics are enabled
  (``detectors.dead_time.enabled == False`` and
  ``detectors.afterpulse.enabled == False``). It is byte-for-byte the V0.1
  chunk engine: bit-identical output, same RNG draw order. V0.1 regression is
  guaranteed by keeping this code path untouched.

* **V0.2 stateful sequential path** — used when dead time and/or afterpulsing
  are enabled. Detector state (READY/DEAD), the afterpulse event queue, the
  rate estimator and all counters persist **continuously across pulses,
  chunks, telemetry windows and checkpoints** — a chunk boundary never resets
  physical state. Within a chunk, candidates (signal, dark, afterpulse) are
  processed in chronological order through the detector state machine.

Physical pipeline (per chunk):

    Alice/source -> photon generation (Poisson) -> channel (loss, delay,
    misalignment) -> Bob (basis selection, PBS routing) -> detector process
    (availability -> efficiency / dark / afterpulse -> avalanche -> dead time)
    -> detection events -> raw telemetry -> windows -> features.

RNG discipline: each chunk uses ``make_chunk_rng(run_seed, chunk_index)``.
The events of chunk *k* are a deterministic function of that chunk's RNG
stream, the configuration, and the *detector state entering chunk k* — which
is itself a deterministic function of the committed earlier chunks. Combined
with checkpointed runtime state, crash recovery stays exact and bounded.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from qism.attacks.base import AttackEngine
from qism.channel.loss import ChannelLoss
from qism.channel.polarization import PolarizationModel
from qism.channel.timing import ChannelTiming
from qism.config import QKDConfig
from qism.environment.base import Environment, make_environment
from qism.random import (
    derive_chunk_seed,
    make_attack_rng,
    make_chunk_rng,
    make_environment_rng,
    make_rng,
    make_sensor_rng,
)
from qism.receiver.bob import Bob
from qism.source.alice import Alice
from qism.source.decoy_state import DecoyStateSource
from qism.source.photon_source import PhotonSource
from qism.telemetry.events import events_frame

log = logging.getLogger("qism.simulator")

INTENSITY_NAMES = np.array(["signal", "decoy", "vacuum"], dtype=object)

#: chronological priority when two candidates share an arrival time
_KIND_RANK = {"afterpulse": 0, "dark": 1, "background": 1, "signal": 2}


def _click_type_label(origin0: str | None, origin1: str | None) -> str:
    """Pulse-level click label from per-detector origins ('none'|origin)."""
    if origin0 is None and origin1 is None:
        return "none"
    if origin0 is not None and origin1 is None:
        return f"d0_{origin0}"
    if origin1 is not None and origin0 is None:
        return f"d1_{origin1}"
    if origin0 == origin1:
        return f"double_{origin0}"
    ap = "afterpulse" in (origin0, origin1)
    if ap:
        return "double_mixed_afterpulse"
    bg = "background" in (origin0, origin1)
    if bg:
        return "double_mixed_background"
    return "double_mixed"


class Simulator:
    """Chunked Monte Carlo event simulator for one run."""

    def __init__(self, config: QKDConfig, run_id: str, seed: int, scenario_id: str | None = None):
        self.config = config
        self.run_id = run_id
        self.seed = int(seed)
        self.scenario_id = scenario_id or config.simulation.scenario_id

        if config.environment.type not in ("stable", "dynamic"):
            raise NotImplementedError(
                f"Environment type '{config.environment.type}' is not supported."
            )

        # --- components ---
        self.alice = Alice(config.source)
        self.decoy = DecoyStateSource(config.source)
        self.photon_source = PhotonSource()
        self.channel_loss = ChannelLoss(config.channel)
        self.channel_timing = ChannelTiming(config.channel)
        self.polarization = PolarizationModel(config.channel)
        self.bob = Bob(config.receiver, config.detectors, self.polarization.p_mis)
        self.attack_engine = AttackEngine(config.attack)
        self.attacks_enabled = config.attack.enabled
        self.environment: Environment = make_environment(config.environment)
        self.environment_enabled = config.environment.enabled or (config.environment.type == "dynamic")
        self.is_dynamic_env = self.environment_enabled and (config.environment.type == "dynamic")
        self.env_state = self.environment.get_state(0.0)

        # --- constants for this run ---
        self.period_s = config.protocol.repetition_period_s
        self.eta_ch = self.channel_loss.transmittance()
        self.channel_delay_s = self.channel_timing.channel_delay_s
        self.p_mis = self.polarization.p_mis
        self.gate_width_s = config.receiver.gate_width_s
        self.window_size = config.telemetry.window_size
        self.chunk_size = config.simulation.chunk_size

        # --- engine selection ---
        dac = config.detectors
        self.dynamics_enabled = dac.dynamics_enabled or self.is_dynamic_env
        self.rate_window_pulses = dac.dead_time.rate_estimation_window_pulses

        # stateful-engine bookkeeping (only meaningful when dynamics enabled)
        self._rate_cand_counts = [0, 0]
        self._rate_window_start = 0  # absolute pulse where the open window began
        if self.dynamics_enabled:
            self._reset_stateful_bookkeeping()

    # ------------------------------------------------------------------ #
    # Runtime state (persistence across chunks / checkpoints)
    # ------------------------------------------------------------------ #
    def _reset_stateful_bookkeeping(self) -> None:
        for det in self.bob.detectors:
            det.reset_runtime()
        self._rate_cand_counts = [0, 0]
        self._rate_window_start = 0

    def runtime_state(self) -> dict[str, Any]:
        """Serialize the full physical runtime state of the simulator.

        This is what makes resumption exact: detector state machines, the
        pending afterpulse queues, environmental drift processes/sensors, and
        the open rate-estimation window are all part of the physics and must
        survive checkpoints.
        """
        state: dict[str, Any] = {
            "detectors": [det.runtime_state() for det in self.bob.detectors],
            "rate_cand_counts": [int(c) for c in self._rate_cand_counts],
            "rate_window_start": int(self._rate_window_start),
        }
        if self.attacks_enabled:
            state["attack_engine"] = self.attack_engine.get_state()
        if hasattr(self.environment, "get_runtime_state"):
            state["environment"] = self.environment.get_runtime_state()
        return state

    def reset_runtime_state(self) -> None:
        """Reset every detector, environment, and the rate estimator to their initial state.

        Called when a run starts fresh (a reused manager may carry detector
        state from a previous partial run).
        """
        for det in self.bob.detectors:
            det.reset_runtime()
        self._rate_cand_counts = [0, 0]
        self._rate_window_start = 0
        self.environment = make_environment(self.config.environment)
        self.env_state = self.environment.get_state(0.0)

    def restore_runtime_state(self, state: dict[str, Any]) -> None:
        """Restore a previously captured runtime state (resume)."""
        dets = state.get("detectors", [])
        for i, det in enumerate(self.bob.detectors):
            if i < len(dets):
                det.restore_runtime_state(dets[i])
        self._rate_cand_counts = [int(c) for c in state.get("rate_cand_counts", [0, 0])]
        self._rate_window_start = int(state.get("rate_window_start", 0))
        attack_state = state.get("attack_engine")
        if attack_state is not None and self.attacks_enabled:
            self.attack_engine.restore_state(attack_state)
        env_state = state.get("environment")
        if env_state is not None and hasattr(self.environment, "restore_runtime_state"):
            self.environment.restore_runtime_state(env_state)
            self.env_state = self.environment.get_state(getattr(self.environment, "t", 0.0))

    # ------------------------------------------------------------------ #
    # Chunked (canonical) event generation
    # ------------------------------------------------------------------ #
    def simulate_chunk(self, chunk_index: int, n_pulses: int, start_pulse: int) -> pd.DataFrame:
        """Simulate one chunk of ``n_pulses`` starting at ``start_pulse``.

        ``start_pulse`` must equal ``chunk_index * chunk_size`` (chunk
        boundaries are absolute so that resumed runs reproduce identical
        per-chunk RNG streams). The physical detector state advances
        continuously across chunk calls — it is never reset here.
        """
        if start_pulse != chunk_index * self.chunk_size:
            raise ValueError(
                f"start_pulse {start_pulse} != chunk_index {chunk_index} * chunk_size {self.chunk_size}; "
                "chunk boundaries must be absolute for reproducible recovery."
            )
        rng = make_chunk_rng(self.seed, chunk_index)
        if self.dynamics_enabled:
            return self._simulate_chunk_stateful(chunk_index, n_pulses, start_pulse, rng)
        return self._simulate_chunk_vectorized(chunk_index, n_pulses, start_pulse, rng)

    # ------------------------------------------------------------------ #
    # V0.1 vectorized engine (byte-identical to V0.1; no dynamics)
    # ------------------------------------------------------------------ #
    def _simulate_chunk_vectorized(
        self, chunk_index: int, n_pulses: int, start_pulse: int, rng
    ) -> pd.DataFrame:
        """The untouched V0.1 engine (used when dynamics are disabled)."""
        n = int(n_pulses)
        pulse_ids = np.arange(start_pulse, start_pulse + n, dtype=np.int64)
        t_pulse = pulse_ids.astype(np.float64) * self.period_s

        alice_bits = self.alice.sample_bits(rng, n)
        alice_bases = self.alice.sample_bases(rng, n)
        intensity_idx = self.decoy.sample_intensity_indices(rng, n)
        mu = self.decoy.mus[intensity_idx]
        photon_number = self.photon_source.sample_photon_numbers(rng, mu)
        arriving = rng.binomial(photon_number.astype(np.int64), self.eta_ch).astype(np.int16)
        expected_arrival = t_pulse + self.channel_delay_s

        meas = self.bob.simulate_measurement(
            rng, alice_bases, alice_bits, arriving, expected_arrival
        )

        click0, click1 = meas.click_0, meas.click_1
        sig0, sig1 = meas.signal_0, meas.signal_1
        dark0, dark1 = meas.dark_0, meas.dark_1
        double = click0 & click1

        click_type = np.full(n, "none", dtype=object)
        m = click0 & ~click1
        click_type[m & sig0 & ~dark0] = "d0_signal"
        click_type[m & ~sig0 & dark0] = "d0_dark"
        click_type[m & sig0 & dark0] = "d0_signal_dark"
        m = click1 & ~click0
        click_type[m & sig1 & ~dark1] = "d1_signal"
        click_type[m & ~sig1 & dark1] = "d1_dark"
        click_type[m & sig1 & dark1] = "d1_signal_dark"
        m = double
        if np.any(m):
            sub = click_type[m].copy()  # all "none" at this point
            sub[sig0[m] & sig1[m]] = "double_signal"
            sub[~sig0[m] & ~sig1[m]] = "double_dark"
            sub[sub == "none"] = "double_mixed"  # one signal + one dark
            click_type[m] = sub

        detector_id = np.full(n, np.nan, dtype=np.float64)
        detector_id[click0 & ~click1] = 0.0
        detector_id[click1 & ~click0] = 1.0
        detector_id[double] = -1.0

        arrival = np.where(
            click0,
            expected_arrival + meas.offset_0,
            np.where(click1, expected_arrival + meas.offset_1, np.nan),
        )
        timing_offset = arrival - expected_arrival

        eta0 = self.bob.detectors[0].efficiency
        eta1 = self.bob.detectors[1].efficiency
        rate0 = self.bob.detectors[0].dark_count_rate_hz
        rate1 = self.bob.detectors[1].dark_count_rate_hz
        detector_efficiency = np.full(n, np.nan, dtype=np.float64)
        detector_efficiency[click0 & ~click1] = eta0
        detector_efficiency[click1 & ~click0] = eta1
        detector_efficiency[double] = eta0
        dark_rate_col = np.full(n, np.nan, dtype=np.float64)
        dark_rate_col[click0 & ~click1] = rate0
        dark_rate_col[click1 & ~click0] = rate1
        dark_rate_col[double] = rate0

        sifted = alice_bases == meas.bob_bases
        single = click0 ^ click1
        bob_bit = np.where(click0, 0, np.where(click1, 1, np.nan))
        bit_error = np.where(sifted & single, (bob_bit != alice_bits).astype(np.float64), np.nan)

        data: dict[str, Any] = {
            "pulse_id": pulse_ids,
            "timestamp": t_pulse,
            "alice_bit": alice_bits,
            "alice_basis": alice_bases,
            "intensity_class": INTENSITY_NAMES[intensity_idx],
            "mu": mu,
            "photon_number": photon_number,
            "channel_transmittance": np.full(n, self.eta_ch, dtype=np.float64),
            "channel_delay_s": np.full(n, self.channel_delay_s, dtype=np.float64),
            "polarization_misalignment": np.full(n, self.p_mis, dtype=np.float64),
            "bob_basis": meas.bob_bases,
            "detector_0_click": click0,
            "detector_1_click": click1,
            "double_click": double,
            "click_type": click_type,
            "detector_id": detector_id,
            "expected_arrival_time_s": expected_arrival,
            "arrival_time_s": arrival,
            "timing_offset_s": timing_offset,
            "gate_width_s": np.full(n, self.gate_width_s, dtype=np.float64),
            "detector_efficiency": detector_efficiency,
            "dark_count_rate": dark_rate_col,
            "detector_state": np.full(n, "ready", dtype=object),
            "sifted": sifted,
            "bit_error": bit_error,
            # V0.2 neutral columns (dynamics disabled)
            "detector_0_state": np.full(n, "ready", dtype=object),
            "detector_1_state": np.full(n, "ready", dtype=object),
            "detector_0_dead_time_remaining_s": np.zeros(n, dtype=np.float64),
            "detector_1_dead_time_remaining_s": np.zeros(n, dtype=np.float64),
            "detector_0_time_since_last_click_s": np.full(n, np.nan, dtype=np.float64),
            "detector_1_time_since_last_click_s": np.full(n, np.nan, dtype=np.float64),
            "detector_0_dead_duration_s": np.full(n, np.nan, dtype=np.float64),
            "detector_1_dead_duration_s": np.full(n, np.nan, dtype=np.float64),
            "detector_0_arrival_time_s": np.where(click0, arrival, np.nan),
            "detector_1_arrival_time_s": np.where(click1 & ~double, arrival, np.nan),
            "dead_time_loss": np.zeros(n, dtype=bool),
            "detector_0_candidate_count": np.zeros(n, dtype=np.int16),
            "detector_1_candidate_count": np.zeros(n, dtype=np.int16),
            "detector_0_dead_rejected_count": np.zeros(n, dtype=np.int16),
            "detector_1_dead_rejected_count": np.zeros(n, dtype=np.int16),
            "afterpulse_candidate": np.zeros(n, dtype=bool),
            "afterpulse_detected": np.zeros(n, dtype=bool),
            "afterpulse_parent_pulse_id": np.full(n, np.nan, dtype=np.float64),
            "afterpulse_delay_s": np.full(n, np.nan, dtype=np.float64),
            # V0.4 infrastructure & environmental telemetry (neutral defaults)
            "sensor_temperature_c": np.full(n, np.nan, dtype=np.float64),
            "sensor_humidity_pct": np.full(n, np.nan, dtype=np.float64),
            "sensor_vibration_g": np.full(n, np.nan, dtype=np.float64),
            "sensor_background_light_lux": np.full(n, np.nan, dtype=np.float64),
            "clock_offset_s": np.zeros(n, dtype=np.float64),
            "true_temperature_c": np.full(n, np.nan, dtype=np.float64),
            "true_humidity_pct": np.full(n, np.nan, dtype=np.float64),
            "true_vibration_g": np.full(n, np.nan, dtype=np.float64),
            "true_background_photon_rate_hz": np.zeros(n, dtype=np.float64),
            "true_channel_loss_db": np.full(n, self.channel_loss.total_loss_db, dtype=np.float64),
            "true_polarization_drift_deg": np.full(n, self.config.channel.misalignment_deg, dtype=np.float64),
            "background_detected": np.zeros(n, dtype=bool),
        }
        env_dict = self.env_state.as_dict()
        if env_dict.get("environment_state") != "stable":
            data["environment_state"] = np.full(n, env_dict["environment_state"], dtype=object)
            for key in ("temperature_c", "bias_voltage_v", "optical_power_dbm"):
                if env_dict.get(key) is not None:
                    data[key] = np.full(n, env_dict[key], dtype=np.float64)

        # --- V0.3 attack integration (vectorized) ---
        if self.attacks_enabled:
            data = self._apply_attacks_vectorized(
                data,
                pulse_ids,
                n,
                chunk_index,
                arrival,
                expected_arrival,
                photon_number,
                alice_bits,
                alice_bases,
                meas.bob_bases,
            )

        return events_frame(data, self.run_id, self.scenario_id, self.window_size)

    # ------------------------------------------------------------------ #
    # V0.3 attack application for vectorized engine
    # ------------------------------------------------------------------ #
    def _apply_attacks_vectorized(
        self,
        data: dict[str, Any],
        pulse_ids: np.ndarray,
        n: int,
        chunk_index: int,
        arrival: np.ndarray,
        expected_arrival: np.ndarray,
        photon_number: np.ndarray,
        alice_bits: np.ndarray,
        alice_bases: np.ndarray,
        bob_bases: np.ndarray,
    ) -> dict[str, Any]:
        """Apply V0.3 causal attacks to vectorized chunk data."""
        o_attack_active = np.zeros(n, dtype=bool)
        o_attack_type = np.full(n, "none", dtype=object)
        o_attack_strength = np.zeros(n, dtype=np.float64)
        o_attack_id: list[str] = ["none"] * n
        o_attack_phase = np.full(n, "normal", dtype=object)

        attack_rng = make_attack_rng(self.seed, chunk_index)

        for i in range(n):
            pulse = int(pulse_ids[i])
            self.attack_engine.update(pulse)
            if self.attack_engine.active_attacks:
                dominant = self.attack_engine.get_dominant_attack()
                atk_phase = self.attack_engine.get_phase()
                event_state: dict[str, Any] = {
                    "pulse_id": pulse,
                    "alice_bit": int(alice_bits[i]),
                    "alice_basis": int(alice_bases[i]),
                    "bob_basis": int(bob_bases[i]),
                    "timestamp": float(data["timestamp"][i]),
                    "mu": float(data["mu"][i]),
                    "photon_number": int(data["photon_number"][i]),
                    "channel_transmittance": float(data["channel_transmittance"][i]),
                    "detector_efficiency": float(
                        data["detector_efficiency"][i]
                        if not np.isnan(data["detector_efficiency"][i])
                        else 0.15
                    ),
                    "arrival_time_s": (
                        float(arrival[i]) if not np.isnan(arrival[i]) else float(expected_arrival[i])
                    ),
                    "arrival_time_offset": 0.0,
                }
                event_state = self.attack_engine.apply(event_state, attack_rng)

                # Source manipulation
                if event_state.get("source_manipulated", False):
                    data["mu"][i] = event_state["mu"]
                    data["photon_number"][i] = int(attack_rng.poisson(data["mu"][i]))
                elif "photon_number" in event_state:
                    data["photon_number"][i] = event_state["photon_number"]

                # Channel loss
                if "channel_transmittance" in event_state:
                    data["channel_transmittance"][i] = event_state["channel_transmittance"]

                # Timing attack: shift arrival time and timing offset
                offset = float(event_state.get("arrival_time_offset", 0.0))
                if offset != 0.0:
                    half_gate = self.gate_width_s / 2.0
                    if not np.isnan(data["timing_offset_s"][i]):
                        new_off = data["timing_offset_s"][i] + offset
                        if abs(new_off) <= half_gate:
                            data["timing_offset_s"][i] = new_off
                            data["arrival_time_s"][i] += offset
                        else:
                            # Displaced outside gate: lost click
                            data["detector_0_click"][i] = False
                            data["detector_1_click"][i] = False
                            data["double_click"][i] = False
                            data["arrival_time_s"][i] = np.nan
                            data["timing_offset_s"][i] = np.nan
                            data["detector_id"][i] = np.nan
                            data["click_type"][i] = "none"
                            data["bit_error"][i] = np.nan

                # Detector manipulation
                if event_state.get("detector_manipulated", False):
                    if not np.isnan(data["detector_efficiency"][i]):
                        if data["detector_id"][i] == 0.0:
                            data["detector_efficiency"][i] = float(event_state.get("detector_0_efficiency", data["detector_efficiency"][i]))
                        elif data["detector_id"][i] == 1.0:
                            data["detector_efficiency"][i] = float(event_state.get("detector_1_efficiency", data["detector_efficiency"][i]))

                if dominant is not None:
                    o_attack_active[i] = True
                    o_attack_type[i] = dominant.scenario.attack_type.value
                    o_attack_strength[i] = dominant.activation_level
                    o_attack_id[i] = dominant.scenario.scenario_id
                    o_attack_phase[i] = atk_phase.value

        data["attack_active"] = o_attack_active
        data["attack_type"] = o_attack_type
        data["attack_strength"] = o_attack_strength
        data["attack_id"] = o_attack_id
        data["attack_phase"] = o_attack_phase
        return data

    # ------------------------------------------------------------------ #
    # V0.2 stateful sequential engine
    # ------------------------------------------------------------------ #
    def _simulate_chunk_stateful(
        self, chunk_index: int, n_pulses: int, start_pulse: int, rng
    ) -> pd.DataFrame:
        """Chronological per-pulse simulation through the detector state
        machines. Source/channel sampling is vectorized; the detector decision
        is causal and sequential. Detector state persists across chunks."""
        n = int(n_pulses)
        half_gate = self.gate_width_s / 2.0
        d0, d1 = self.bob.detectors[0], self.bob.detectors[1]
        p_dark0 = d0.dark_click_probability()
        p_dark1 = d1.dark_click_probability()
        dets = (d0, d1)

        # ---------- vectorized source / channel / routing ----------
        pulse_ids = np.arange(start_pulse, start_pulse + n, dtype=np.int64)
        t_pulse = pulse_ids.astype(np.float64) * self.period_s
        alice_bits = self.alice.sample_bits(rng, n)
        alice_bases = self.alice.sample_bases(rng, n)
        intensity_idx = self.decoy.sample_intensity_indices(rng, n)
        mu = self.decoy.mus[intensity_idx].copy()
        photon_number = self.photon_source.sample_photon_numbers(rng, mu)
        arriving = rng.binomial(photon_number.astype(np.int64), self.eta_ch).astype(np.int64)
        bob_bases = self.bob.sample_bases(rng, n)
        intended = self.bob.pbs.intended_detector(rng, alice_bases, bob_bases, alice_bits)
        expected_arrival = t_pulse + self.channel_delay_s
        # routed photon number per detector per pulse
        n0 = np.where(intended == 0, arriving, 0)
        n1 = np.where(intended == 1, arriving, 0)

        # dark-count presence + arrival offset (drawn for every pulse)
        dark_pres = (
            (rng.random(n) < p_dark0).astype(bool),
            (rng.random(n) < p_dark1).astype(bool),
        )
        dark_off = (
            rng.uniform(-half_gate, half_gate, n),
            rng.uniform(-half_gate, half_gate, n),
        )

        # ---------- output arrays ----------
        o_click0 = np.zeros(n, dtype=bool)
        o_click1 = np.zeros(n, dtype=bool)
        o_origin0 = np.full(n, "", dtype=object)  # "" = no click
        o_origin1 = np.full(n, "", dtype=object)
        o_arr0 = np.full(n, np.nan, dtype=np.float64)
        o_arr1 = np.full(n, np.nan, dtype=np.float64)
        o_dead_dur0 = np.full(n, np.nan, dtype=np.float64)
        o_dead_dur1 = np.full(n, np.nan, dtype=np.float64)
        o_st0 = np.full(n, "ready", dtype=object)
        o_st1 = np.full(n, "ready", dtype=object)
        o_rem0 = np.zeros(n, dtype=np.float64)
        o_rem1 = np.zeros(n, dtype=np.float64)
        o_since0 = np.full(n, np.nan, dtype=np.float64)
        o_since1 = np.full(n, np.nan, dtype=np.float64)
        o_cand0 = np.zeros(n, dtype=np.int16)
        o_cand1 = np.zeros(n, dtype=np.int16)
        o_rej0 = np.zeros(n, dtype=np.int16)
        o_rej1 = np.zeros(n, dtype=np.int16)
        o_ap_cand = np.zeros(n, dtype=bool)
        o_ap_det = np.zeros(n, dtype=bool)
        o_ap_parent = np.full(n, np.nan, dtype=np.float64)
        o_ap_delay = np.full(n, np.nan, dtype=np.float64)
        o_dead_loss = np.zeros(n, dtype=bool)

        w = self.rate_window_pulses  # rate-estimation window (absolute pulses)

        # V0.4 dynamic environment setup
        is_dyn = self.is_dynamic_env and hasattr(self.environment, "step_dynamics")
        env_rng = make_environment_rng(derive_chunk_seed(self.seed, chunk_index)) if is_dyn else None
        sensor_rng = make_sensor_rng(derive_chunk_seed(self.seed, chunk_index)) if is_dyn else None

        o_sensor_temp = np.full(n, np.nan, dtype=np.float64)
        o_sensor_hum = np.full(n, np.nan, dtype=np.float64)
        o_sensor_vib = np.full(n, np.nan, dtype=np.float64)
        o_sensor_bg = np.full(n, np.nan, dtype=np.float64)
        o_clock_offset = np.zeros(n, dtype=np.float64)
        o_true_temp = np.full(n, np.nan, dtype=np.float64)
        o_true_hum = np.full(n, np.nan, dtype=np.float64)
        o_true_vib = np.full(n, np.nan, dtype=np.float64)
        o_true_bg_rate = np.zeros(n, dtype=np.float64)
        o_true_loss_db = np.full(n, self.channel_loss.total_loss_db, dtype=np.float64)
        o_true_pol_deg = np.full(n, self.config.channel.misalignment_deg, dtype=np.float64)
        o_bg_det = np.zeros(n, dtype=bool)

        # V0.3 attack output columns (initialized here, filled in the loop)
        o_attack_active = np.zeros(n, dtype=bool)
        o_attack_type = np.full(n, "none", dtype=object)
        o_attack_strength = np.zeros(n, dtype=np.float64)
        o_attack_phase = np.full(n, "normal", dtype=object)
        o_attack_id_arr: list[str] = ["none"] * n
        o_channel_eta = np.full(n, self.eta_ch, dtype=np.float64)
        o_eff0 = np.full(n, d0.efficiency, dtype=np.float64)
        o_eff1 = np.full(n, d1.efficiency, dtype=np.float64)
        o_dark_rate0 = np.full(n, d0.dark_count_rate_hz, dtype=np.float64)
        o_dark_rate1 = np.full(n, d1.dark_count_rate_hz, dtype=np.float64)

        attack_rng = make_attack_rng(self.seed, chunk_index) if self.attacks_enabled else None

        for i in range(n):
            t_exp = expected_arrival[i]
            t_exp_pulse = t_exp
            eff_d0 = d0.efficiency
            eff_d1 = d1.efficiency
            extra_dark0 = 0.0
            extra_dark1 = 0.0

            # ---- V0.4 dynamic environment update for this pulse ----
            if is_dyn and env_rng is not None and sensor_rng is not None:
                cur_env = self.environment.step_dynamics(self.period_s, env_rng, sensor_rng)
                self.env_state = cur_env

                temp_c = cur_env.temperature_c
                o_sensor_temp[i] = cur_env.sensor_temperature_c
                o_sensor_hum[i] = cur_env.sensor_humidity_pct
                o_sensor_vib[i] = cur_env.sensor_vibration_g
                o_sensor_bg[i] = cur_env.sensor_background_light_lux
                o_clock_offset[i] = cur_env.clock_offset_s
                o_true_temp[i] = temp_c
                o_true_hum[i] = cur_env.humidity_pct
                o_true_vib[i] = cur_env.vibration_g
                o_true_bg_rate[i] = cur_env.background_photon_rate_hz
                o_true_loss_db[i] = self.channel_loss.total_loss_db + cur_env.channel_loss_drift_db
                o_true_pol_deg[i] = self.config.channel.misalignment_deg + cur_env.polarization_drift_deg

                # 1. Dynamic source intensity drift & pulse jitter
                delta_src = cur_env.source_intensity_drift
                if delta_src != 0.0:
                    mu[i] = max(0.0, mu[i] * (1.0 + delta_src))
                if self.config.source.intensity_jitter_std > 0.0:
                    mu[i] = max(0.0, mu[i] * (1.0 + float(rng.normal(0.0, self.config.source.intensity_jitter_std))))
                if delta_src != 0.0 or self.config.source.intensity_jitter_std > 0.0:
                    photon_number[i] = int(rng.poisson(mu[i]))

                # 2. Dynamic channel loss drift
                dyn_loss_db = cur_env.channel_loss_drift_db
                dyn_eta = max(0.0, min(1.0, self.channel_loss.transmittance(dyn_loss_db)))
                o_channel_eta[i] = dyn_eta
                if dyn_loss_db != 0.0:
                    arriving[i] = int(rng.binomial(int(photon_number[i]), dyn_eta))

                # 3. Dynamic polarization drift
                dyn_pol_deg = cur_env.polarization_drift_deg
                if dyn_pol_deg != 0.0:
                    total_pol = self.config.channel.misalignment_deg + dyn_pol_deg
                    p_mis_dyn = self.polarization.p_mis_at_angle(total_pol)
                    p_same = 1.0 - p_mis_dyn if alice_bases[i] == bob_bases[i] else 0.5
                    p_d0 = p_same if alice_bits[i] == 0 else 1.0 - p_same
                    int_det = 0 if rng.random() < p_d0 else 1
                    n0[i] = arriving[i] if int_det == 0 else 0
                    n1[i] = arriving[i] if int_det == 1 else 0

                # 4. Detector efficiency & dark rate
                eff_d0 = max(0.0, min(1.0, d0.efficiency_at_temperature(temp_c) * (1.0 + cur_env.detector_efficiency_drift)))
                eff_d1 = max(0.0, min(1.0, d1.efficiency_at_temperature(temp_c) * (1.0 + cur_env.detector_efficiency_drift)))
                o_eff0[i] = eff_d0
                o_eff1[i] = eff_d1

                dark_rate0 = d0.dark_rate_at_temperature(temp_c)
                dark_rate1 = d1.dark_rate_at_temperature(temp_c)
                o_dark_rate0[i] = dark_rate0
                o_dark_rate1[i] = dark_rate1

            # ---- V0.3: update attack state for this pulse ----
            if self.attacks_enabled and attack_rng is not None:
                self.attack_engine.update(int(pulse_ids[i]))
                if self.attack_engine.active_attacks:
                    dominant = self.attack_engine.get_dominant_attack()
                    atk_phase = self.attack_engine.get_phase()
                    atk_state: dict[str, Any] = {
                        "pulse_id": int(pulse_ids[i]),
                        "mu": float(mu[i]),
                        "photon_number": int(photon_number[i]),
                        "channel_transmittance": float(o_channel_eta[i]),
                        "arrival_time_s": float(t_exp),
                        "alice_basis": int(alice_bases[i]),
                        "alice_bit": int(alice_bits[i]),
                        "bob_basis": int(bob_bases[i]),
                        "detector_0_efficiency": float(eff_d0),
                        "detector_1_efficiency": float(eff_d1),
                        "detector_0_dark_rate": float(o_dark_rate0[i]),
                        "detector_1_dark_rate": float(o_dark_rate1[i]),
                        "arrival_time_offset": 0.0,
                    }
                    atk_state = self.attack_engine.apply(atk_state, attack_rng)

                    # 1. Source manipulation
                    if atk_state.get("source_manipulated", False):
                        mu[i] = atk_state["mu"]
                        photon_number[i] = int(attack_rng.poisson(mu[i]))
                    elif "photon_number" in atk_state:
                        photon_number[i] = int(atk_state["photon_number"])

                    # 2. Channel transmittance & arriving photons
                    new_eta = max(0.0, min(1.0, float(atk_state.get("channel_transmittance", o_channel_eta[i]))))
                    o_channel_eta[i] = new_eta
                    n_arr = int(attack_rng.binomial(int(photon_number[i]), new_eta))
                    arriving[i] = n_arr

                    # 3. Intercept-Resend PBS routing
                    if atk_state.get("attack_resend", False):
                        eve_basis = int(atk_state["eve_basis"])
                        eve_bit = int(atk_state["eve_bit"])
                        bob_b = int(bob_bases[i])
                        if bob_b == eve_basis:
                            # Deterministic routing to detector eve_bit
                            n0[i] = n_arr if eve_bit == 0 else 0
                            n1[i] = n_arr if eve_bit == 1 else 0
                        else:
                            # 50/50 split at Bob's PBS
                            routed_0 = int(attack_rng.binomial(n_arr, 0.5))
                            n0[i] = routed_0
                            n1[i] = n_arr - routed_0
                    else:
                        n0[i] = n_arr if intended[i] == 0 else 0
                        n1[i] = n_arr if intended[i] == 1 else 0

                    # 4. Timing attack
                    offset = float(atk_state.get("arrival_time_offset", 0.0))
                    t_exp_pulse = t_exp + offset

                    # 5. Detector manipulation
                    eff_d0 = float(atk_state.get("detector_0_efficiency", eff_d0))
                    eff_d1 = float(atk_state.get("detector_1_efficiency", eff_d1))
                    extra_dark0 = max(0.0, float(atk_state.get("detector_0_dark_rate", o_dark_rate0[i])) - o_dark_rate0[i])
                    extra_dark1 = max(0.0, float(atk_state.get("detector_1_dark_rate", o_dark_rate1[i])) - o_dark_rate1[i])

                    o_eff0[i] = eff_d0
                    o_eff1[i] = eff_d1
                    o_dark_rate0[i] = o_dark_rate0[i] + extra_dark0
                    o_dark_rate1[i] = o_dark_rate1[i] + extra_dark1

                    # Record attack metadata
                    if dominant is not None:
                        o_attack_active[i] = True
                        o_attack_type[i] = dominant.scenario.attack_type.value
                        o_attack_strength[i] = dominant.activation_level
                        o_attack_id_arr[i] = dominant.scenario.scenario_id
                        o_attack_phase[i] = atk_phase.value
                else:
                    o_attack_phase[i] = self.attack_engine.get_phase().value

            g_start = t_exp - half_gate
            g_end = t_exp + half_gate

            # ---------------- per-detector candidate phase ----------------
            row = {}
            for di, det in enumerate(dets):
                # drain afterpulse candidates due at/before this gate end
                due = det.pop_pending_until(g_end)
                in_gate: list[tuple[float, str, Any]] = []
                ap_first: dict[str, Any] | None = None  # first attempted (any)
                ap_clicked: dict[str, Any] | None = None
                for entry in due:
                    if ap_first is None:
                        ap_first = {
                            "parent": float(entry.parent_pulse_id),
                            "delay": float(entry.delay_s),
                        }
                    if entry.t < g_start:
                        # between gates: detector is not biased -> suppressed
                        continue
                    in_gate.append((entry.t, "afterpulse", entry))

                # signal candidate (photons inside the gate, jittered arrival)
                nn_sig = int(n0[i]) if di == 0 else int(n1[i])
                eff_actual = eff_d0 if di == 0 else eff_d1
                p_click_sig = 0.0
                if nn_sig > 0:
                    eps = float(rng.normal(0.0, det.timing_jitter_s))
                    t_sig = t_exp_pulse + eps
                    if g_start <= t_sig <= g_end:
                        p_click_sig = float(-np.expm1(nn_sig * np.log1p(-eff_actual)))
                        in_gate.append((t_sig, "signal", None))

                # dark candidate (intrinsic arrival inside the gate)
                if dark_pres[di][i]:
                    suppress = False
                    if is_dyn and d0.temp_coeff_dark_count != 0.0 and env_rng is not None:
                        d_rate_t = dark_rate0 if di == 0 else dark_rate1
                        d_base = d0.dark_count_rate_hz if di == 0 else d1.dark_count_rate_hz
                        if d_rate_t < d_base and env_rng.random() >= (d_rate_t / max(1e-12, d_base)):
                            suppress = True
                    if not suppress:
                        in_gate.append((t_exp + dark_off[di][i], "dark", None))

                # Extra dark counts from temperature scaling up
                if is_dyn and d0.temp_coeff_dark_count != 0.0 and env_rng is not None:
                    d_rate_t = dark_rate0 if di == 0 else dark_rate1
                    d_base = d0.dark_count_rate_hz if di == 0 else d1.dark_count_rate_hz
                    if d_rate_t > d_base:
                        p_extra_t = (d_rate_t - d_base) * self.gate_width_s
                        if env_rng.random() < p_extra_t:
                            t_dark_extra = t_exp + float(env_rng.uniform(-half_gate, half_gate))
                            in_gate.append((t_dark_extra, "dark", None))

                extra_dark_rate = extra_dark0 if di == 0 else extra_dark1
                if extra_dark_rate > 0.0 and attack_rng is not None:
                    p_extra = extra_dark_rate * self.gate_width_s
                    if attack_rng.random() < p_extra:
                        t_dark_extra = t_exp + float(attack_rng.uniform(-half_gate, half_gate))
                        in_gate.append((t_dark_extra, "dark", None))

                # background stray photon candidate (optical noise inside gate)
                if is_dyn and self.config.background.enabled and cur_env.background_photon_rate_hz > 0.0 and env_rng is not None:
                    # 50% split of stray background flux between detectors
                    p_bg = 1.0 - math.exp(-0.5 * cur_env.background_photon_rate_hz * self.gate_width_s)
                    if env_rng.random() < p_bg:
                        t_bg = float(env_rng.uniform(g_start, g_end))
                        in_gate.append((t_bg, "background", None))

                in_gate.sort(key=lambda c: (c[0], _KIND_RANK[c[1]]))

                # detector state at the gate (before in-gate candidates)
                det.update_state(g_start)
                if di == 0:
                    o_st0[i] = det.state.value
                    o_rem0[i] = det.dead_time_remaining(g_start)
                    o_since0[i] = det.time_since_last_click(g_start)
                else:
                    o_st1[i] = det.state.value
                    o_rem1[i] = det.dead_time_remaining(g_start)
                    o_since1[i] = det.time_since_last_click(g_start)

                cand_count = 0
                rej_count = 0
                clicked_t: float | None = None
                clicked_origin: str | None = None
                dead_dur = 0.0
                if ap_first is not None:
                    o_ap_cand[i] = True
                for t_cand, kind, entry in in_gate:
                    cand_count += 1
                    if kind == "afterpulse":
                        # p_click=1: an afterpulse that finds the detector READY
                        # avalanches (gated interpretation: it is in-gate).
                        res = det.offer(
                            t_cand,
                            rng,
                            kind="afterpulse",
                            p_click=1.0,
                            parent_pulse_id=entry.parent_pulse_id,
                            afterpulse_delay_s=entry.delay_s,
                        )
                        if ap_clicked is None and res.clicked:
                            ap_clicked = {
                                "parent": float(entry.parent_pulse_id),
                                "delay": float(entry.delay_s),
                            }
                    else:
                        p = p_click_sig if kind == "signal" else (eff_actual if kind == "background" else 1.0)
                        res = det.offer(t_cand, rng, kind=kind, p_click=p)
                    if not res.available:
                        rej_count += 1
                    if clicked_t is None and res.clicked:
                        clicked_t = res.t
                        clicked_origin = res.origin
                        dead_dur = res.dead_duration
                        if kind == "background":
                            o_bg_det[i] = True
                # newly scheduled afterpulses that land in this same gate
                while True:
                    nxt = det.pending_front()
                    if nxt is None or nxt.t > g_end:
                        break
                    det.pop_next_pending()
                    cand_count += 1
                    res = det.offer(
                        nxt.t,
                        rng,
                        kind="afterpulse",
                        p_click=1.0,
                        parent_pulse_id=nxt.parent_pulse_id,
                        afterpulse_delay_s=nxt.delay_s,
                    )
                    if not res.available:
                        rej_count += 1
                    if ap_clicked is None and res.clicked:
                        ap_clicked = {
                            "parent": float(nxt.parent_pulse_id),
                            "delay": float(nxt.delay_s),
                        }
                    if clicked_t is None and res.clicked:
                        clicked_t = res.t
                        clicked_origin = res.origin
                        dead_dur = res.dead_duration

                if di == 0:
                    o_click0[i] = clicked_t is not None
                    o_origin0[i] = clicked_origin or ""
                    if clicked_t is not None:
                        o_arr0[i] = clicked_t
                        o_dead_dur0[i] = dead_dur
                    o_cand0[i] = cand_count
                    o_rej0[i] = rej_count
                    row["ap0"] = ap_first, ap_clicked
                else:
                    o_click1[i] = clicked_t is not None
                    o_origin1[i] = clicked_origin or ""
                    if clicked_t is not None:
                        o_arr1[i] = clicked_t
                        o_dead_dur1[i] = dead_dur
                    o_cand1[i] = cand_count
                    o_rej1[i] = rej_count
                    row["ap1"] = ap_first, ap_clicked

            # combine the two detectors' afterpulse attribution for this pulse
            ap0_first, ap0_det = row["ap0"]
            ap1_first, ap1_det = row["ap1"]
            first = ap0_first or ap1_first
            det_ev = ap0_det or ap1_det
            if first is not None:
                o_ap_cand[i] = True
                o_ap_parent[i] = first["parent"]
                o_ap_delay[i] = first["delay"]
            if det_ev is not None:
                o_ap_det[i] = True
            o_dead_loss[i] = o_rej0[i] > 0 or o_rej1[i] > 0

            # ---- rate-estimation window accounting (per detector) ----
            for di, det in enumerate(dets):
                self._rate_cand_counts[di] += o_cand0[i] if di == 0 else o_cand1[i]
            boundary = start_pulse + i + 1
            if boundary % w == 0:
                span = (boundary - self._rate_window_start) * self.period_s
                for di, det in enumerate(dets):
                    lam = self._rate_cand_counts[di] / span if span > 0 else 0.0
                    det.set_candidate_rate(lam)
                    self._rate_cand_counts[di] = 0
                self._rate_window_start = boundary

        # ---------- assemble event rows ----------
        click0, click1 = o_click0, o_click1
        double = click0 & click1
        click_type = np.empty(n, dtype=object)
        for i in range(n):
            origin0 = o_origin0[i] or None
            origin1 = o_origin1[i] or None
            click_type[i] = _click_type_label(origin0, origin1)

        detector_id = np.full(n, np.nan, dtype=np.float64)
        detector_id[click0 & ~click1] = 0.0
        detector_id[click1 & ~click0] = 1.0
        detector_id[double] = -1.0

        arrival = np.where(click0, o_arr0, np.where(click1, o_arr1, np.nan))
        if is_dyn:
            arrival = arrival + o_clock_offset
            o_arr0 = np.where(click0, o_arr0 + o_clock_offset, np.nan)
            o_arr1 = np.where(click1, o_arr1 + o_clock_offset, np.nan)
        timing_offset = arrival - expected_arrival

        detector_efficiency = np.full(n, np.nan, dtype=np.float64)
        detector_efficiency[click0] = o_eff0[click0]
        detector_efficiency[click1 & ~click0] = o_eff1[click1 & ~click0]
        dark_rate_col = np.full(n, np.nan, dtype=np.float64)
        dark_rate_col[click0] = o_dark_rate0[click0]
        dark_rate_col[click1 & ~click0] = o_dark_rate1[click1 & ~click0]

        # legacy detector_state column: clicked detector was READY; for
        # no-click pulses report 'dead' only when both detectors were DEAD.
        detector_state = np.full(n, "ready", dtype=object)
        no_click = ~(click0 | click1)
        both_dead = no_click & (o_st0 == "dead") & (o_st1 == "dead")
        detector_state[both_dead] = "dead"

        sifted = alice_bases == bob_bases
        single = click0 ^ click1
        bob_bit = np.where(click0, 0, np.where(click1, 1, np.nan))
        bit_error = np.where(sifted & single, (bob_bit != alice_bits).astype(np.float64), np.nan)

        data: dict[str, Any] = {
            "pulse_id": pulse_ids,
            "timestamp": t_pulse,
            "alice_bit": alice_bits,
            "alice_basis": alice_bases,
            "intensity_class": INTENSITY_NAMES[intensity_idx],
            "mu": mu,
            "photon_number": photon_number,
            "channel_transmittance": o_channel_eta,
            "channel_delay_s": np.full(n, self.channel_delay_s, dtype=np.float64),
            "polarization_misalignment": np.full(n, self.p_mis, dtype=np.float64),
            "bob_basis": bob_bases,
            "detector_0_click": click0,
            "detector_1_click": click1,
            "double_click": double,
            "click_type": click_type,
            "detector_id": detector_id,
            "expected_arrival_time_s": expected_arrival,
            "arrival_time_s": arrival,
            "timing_offset_s": timing_offset,
            "gate_width_s": np.full(n, self.gate_width_s, dtype=np.float64),
            "detector_efficiency": detector_efficiency,
            "dark_count_rate": dark_rate_col,
            "detector_state": detector_state,
            "sifted": sifted,
            "bit_error": bit_error,
            # V0.2 detector dynamics
            "detector_0_state": o_st0,
            "detector_1_state": o_st1,
            "detector_0_dead_time_remaining_s": o_rem0,
            "detector_1_dead_time_remaining_s": o_rem1,
            "detector_0_time_since_last_click_s": o_since0,
            "detector_1_time_since_last_click_s": o_since1,
            "detector_0_dead_duration_s": o_dead_dur0,
            "detector_1_dead_duration_s": o_dead_dur1,
            "detector_0_arrival_time_s": o_arr0,
            "detector_1_arrival_time_s": o_arr1,
            "dead_time_loss": o_dead_loss,
            "detector_0_candidate_count": o_cand0,
            "detector_1_candidate_count": o_cand1,
            "detector_0_dead_rejected_count": o_rej0,
            "detector_1_dead_rejected_count": o_rej1,
            "afterpulse_candidate": o_ap_cand,
            "afterpulse_detected": o_ap_det,
            "afterpulse_parent_pulse_id": o_ap_parent,
            "afterpulse_delay_s": o_ap_delay,
            # V0.3 attack metadata
            "attack_active": o_attack_active,
            "attack_type": o_attack_type,
            "attack_strength": o_attack_strength,
            "attack_id": o_attack_id_arr,
            "attack_phase": o_attack_phase,
            # V0.4 infrastructure & environmental telemetry
            "sensor_temperature_c": o_sensor_temp,
            "sensor_humidity_pct": o_sensor_hum,
            "sensor_vibration_g": o_sensor_vib,
            "sensor_background_light_lux": o_sensor_bg,
            "clock_offset_s": o_clock_offset,
            "true_temperature_c": o_true_temp,
            "true_humidity_pct": o_true_hum,
            "true_vibration_g": o_true_vib,
            "true_background_photon_rate_hz": o_true_bg_rate,
            "true_channel_loss_db": o_true_loss_db,
            "true_polarization_drift_deg": o_true_pol_deg,
            "background_detected": o_bg_det,
        }
        env_dict = self.env_state.as_dict()
        if env_dict.get("environment_state") != "stable":
            data["environment_state"] = np.full(n, env_dict["environment_state"], dtype=object)
            for key in ("temperature_c", "bias_voltage_v", "optical_power_dbm"):
                if env_dict.get(key) is not None:
                    data[key] = np.full(n, env_dict[key], dtype=np.float64)

        # bound memory: busy intervals older than this chunk are no longer
        # needed for telemetry (kept only for the interval-union diagnostics)
        for det in dets:
            det.prune_history_before(t_pulse[0])

        return events_frame(data, self.run_id, self.scenario_id, self.window_size)

    # ------------------------------------------------------------------ #
    # Per-pulse API (V0.1 convenience; stateless only)
    # ------------------------------------------------------------------ #
    def simulate_pulse(self, pulse_id: int) -> dict[str, Any]:
        """Simulate a single pulse independently (V0.1 convenience API).

        Statistically equivalent to ``simulate_chunk`` but uses its own RNG
        stream; it is NOT bit-identical to the same pulse inside a full chunk.
        Unavailable when detector dynamics are enabled (stateful simulation
        must run as a continuous stream through ``simulate_chunk``).
        """
        if self.dynamics_enabled:
            raise NotImplementedError(
                "simulate_pulse is a stateless V0.1 convenience API and is not "
                "available when detector dynamics are enabled; simulate "
                "contiguous chunks with simulate_chunk instead."
            )
        pulse_id = int(pulse_id)
        rng = make_rng(derive_chunk_seed(self.seed, pulse_id))
        df = self._simulate_chunk_vectorized(pulse_id, 1, pulse_id, rng)
        row = df.iloc[0].to_dict()
        event_state: dict[str, Any] = dict(row)
        if self.attacks_enabled:
            self.attack_engine.update(pulse_id)
            event_state = self.attack_engine.apply(event_state, rng)
        return event_state
