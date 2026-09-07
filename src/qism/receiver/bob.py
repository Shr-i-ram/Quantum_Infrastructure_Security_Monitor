"""Bob's receiver: two-detector measurement.

Bob has two independent Detector objects (D0, D1) — never a single generic
detection probability. This separation is essential for future detector
imbalance and implementation attacks.

The measurement pipeline per pulse:

    Bob basis -> PBS routing (ideal outcome + misalignment)
    -> per-detector signal clicks (1-(1-eta)^n on the routed detector)
    -> per-detector dark clicks (1-exp(-lambda*Delta_t))
    -> gate acceptance for signal photons
    -> arrival offsets (signal: Gaussian jitter; dark: uniform in gate)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from qism.config import DetectorArrayConfig, DetectorConfig, ReceiverConfig
from qism.detector.detector import Detector
from qism.receiver.basis_selector import BasisSelector
from qism.receiver.pbs import PolarizingBeamSplitter


@dataclass
class MeasurementResult:
    """Result of Bob's measurement over a chunk of pulses (all arrays length n)."""

    bob_bases: np.ndarray
    intended_detector: np.ndarray
    click_0: np.ndarray
    click_1: np.ndarray
    signal_0: np.ndarray
    signal_1: np.ndarray
    dark_0: np.ndarray
    dark_1: np.ndarray
    offset_0: np.ndarray
    offset_1: np.ndarray


class Bob:
    """Bob's receiver: basis selection + PBS + two detectors."""

    def __init__(
        self,
        receiver_config: ReceiverConfig,
        detector_array_config: DetectorArrayConfig,
        misalignment_probability: float,
    ):
        self.receiver_config = receiver_config
        self.detector_array_config = detector_array_config
        self.basis_selector = BasisSelector(receiver_config)
        self.pbs = PolarizingBeamSplitter(misalignment_probability)
        self.detectors: list[Detector] = [
            self._make_detector(0, detector_array_config.detector_0),
            self._make_detector(1, detector_array_config.detector_1),
        ]

    def _make_detector(self, index: int, cfg: DetectorConfig) -> Detector:
        dac = self.detector_array_config
        return Detector(
            config=cfg,
            index=index,
            gate_width_s=self.receiver_config.gate_width_s,
            detection_interval_s=self.receiver_config.effective_detection_interval_s,
            dead_time=dac.dead_time,
            afterpulse=dac.afterpulse,
        )

    def sample_bases(self, rng: Generator, n: int) -> np.ndarray:
        return self.basis_selector.sample(rng, n)

    def simulate_measurement(
        self,
        rng: Generator,
        alice_bases: np.ndarray,
        alice_bits: np.ndarray,
        photon_numbers: np.ndarray,
        t_expected: np.ndarray,
    ) -> MeasurementResult:
        """Run Bob's measurement on a chunk of pulses (vectorized)."""
        n = len(alice_bases)
        bob_bases = self.sample_bases(rng, n)
        d_int = self.pbs.intended_detector(rng, alice_bases, bob_bases, alice_bits)

        d0, d1 = self.detectors
        p_sig0 = np.where(d_int == 0, d0.signal_click_probability(photon_numbers), 0.0)
        p_sig1 = np.where(d_int == 1, d1.signal_click_probability(photon_numbers), 0.0)
        signal_0 = rng.random(n) < p_sig0
        signal_1 = rng.random(n) < p_sig1

        p_dark0 = d0.dark_click_probability()
        p_dark1 = d1.dark_click_probability()
        dark_0 = rng.random(n) < p_dark0
        dark_1 = rng.random(n) < p_dark1

        # Gate acceptance for signal photons (jitter must land inside the gate)
        gate_width = self.receiver_config.gate_width_s
        eps0 = rng.normal(0.0, d0.timing_jitter_s, n)
        eps1 = rng.normal(0.0, d1.timing_jitter_s, n)
        half = gate_width / 2.0
        signal_0 = signal_0 & (np.abs(eps0) <= half)
        signal_1 = signal_1 & (np.abs(eps1) <= half)

        # Dark arrivals are uniform inside the gate
        off_dark0 = rng.uniform(-half, half, n)
        off_dark1 = rng.uniform(-half, half, n)
        offset_0 = np.where(signal_0, eps0, np.where(dark_0, off_dark0, np.nan))
        offset_1 = np.where(signal_1, eps1, np.where(dark_1, off_dark1, np.nan))

        click_0 = signal_0 | dark_0
        click_1 = signal_1 | dark_1

        return MeasurementResult(
            bob_bases=bob_bases,
            intended_detector=d_int,
            click_0=click_0,
            click_1=click_1,
            signal_0=signal_0,
            signal_1=signal_1,
            dark_0=dark_0,
            dark_1=dark_1,
            offset_0=offset_0,
            offset_1=offset_1,
        )
