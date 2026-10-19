"""Live QKD Simulation Service for QISM Live Demonstration Console.

Wraps the physical Monte Carlo QKD Simulator and WindowAggregator.
Provides dynamic attack injection, real-time chunked event generation,
online model inference, and WebSocket telemetry streaming.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable

from ..config import REPO_ROOT, settings
from ..database import log_audit, save_session, save_telemetry_point
from .incident_service import incident_service
from .inference_service import inference_service

# Ensure src is on path for qism imports
src_path = str(REPO_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from qism.attacks.base import AttackScenario, AttackType
from qism.attacks.intercept_resend import (
    ChannelLossAttack,
    DetectorManipulationAttack,
    InterceptResendAttack,
    PNSAttack,
    SourceManipulationAttack,
    TimingAttack,
)
from qism.config import QKDConfig
from qism.simulation.simulator import Simulator
from qism.telemetry.windows import WindowAggregator

log = logging.getLogger("qism.demo.simulator")

ATTACK_MAP = {
    "intercept_resend": (AttackType.INTERCEPT_RESEND, InterceptResendAttack),
    "pns": (AttackType.PNS, PNSAttack),
    "detector_manipulation": (AttackType.DETECTOR_MANIPULATION, DetectorManipulationAttack),
    "channel_loss": (AttackType.CHANNEL_LOSS, ChannelLossAttack),
    "timing": (AttackType.TIMING, TimingAttack),
    "source_manipulation": (AttackType.SOURCE_MANIPULATION, SourceManipulationAttack),
}


class SimulatorService:
    """Manages the lifecycle of a live QKD simulation session."""

    def __init__(self) -> None:
        self.status: str = "idle"  # idle, running, paused, stopped, error
        self.session_id: str | None = None
        self.scenario_id: str = "live_default"
        self.model_id: str = settings.default_model_id
        self.window_index: int = 0
        self.chunk_size: int = 2000  # 2000 pulses per window for fast online demo
        self.active_attack: dict[str, Any] | None = None

        self._simulator: Simulator | None = None
        self._aggregator: WindowAggregator | None = None
        self._task: asyncio.Task | None = None
        self._subscribers: list[Callable[[dict[str, Any]], Any]] = []

    def subscribe(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        """Register a subscriber callback for emitted telemetry windows."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        """Remove a subscriber callback."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    @staticmethod
    def get_supported_attacks() -> list[dict[str, Any]]:
        """Return metadata for all physically supported attack mechanisms."""
        return [
            {
                "id": "intercept_resend",
                "name": "Intercept-Resend Attack",
                "target": "Quantum Channel / Eavesdropper",
                "mechanism": "Eve measures photons in random conjugate bases and resends new photons.",
                "default_severity": 0.8,
                "expected_signature": "QBER increases dramatically (~20-25%), state space shifts away from normal hypersphere.",
                "parameters": [
                    {"name": "intensity", "label": "Intercept Probability", "min": 0.1, "max": 1.0, "default": 0.8}
                ],
            },
            {
                "id": "pns",
                "name": "Photon Number Splitting (PNS)",
                "target": "Source Multiphoton Pulses",
                "mechanism": "Eve splits off one photon from multiphoton pulses and stores in quantum memory.",
                "default_severity": 0.8,
                "expected_signature": "Subtle shift in signal/decoy gain ratios; detectable by multi-feature QML.",
                "parameters": [
                    {"name": "split_ratio", "label": "Splitting Efficiency", "min": 0.1, "max": 1.0, "default": 0.8}
                ],
            },
            {
                "id": "detector_manipulation",
                "name": "Detector Blinding / Manipulation",
                "target": "Bob SPAD Detectors",
                "mechanism": "Eve sends continuous-wave blinding light to force APDs into linear photoconductive mode.",
                "default_severity": 0.8,
                "expected_signature": "Severe detector imbalance and dark count rate distortion.",
                "parameters": [
                    {"name": "blinding_intensity", "label": "Blinding Intensity", "min": 0.1, "max": 1.0, "default": 0.8}
                ],
            },
            {
                "id": "channel_loss",
                "name": "Channel Loss Attack",
                "target": "Optical Fiber Medium",
                "mechanism": "Eve adds physical attenuation or bending loss to mask selective eavesdropping.",
                "default_severity": 0.5,
                "expected_signature": "Drop in overall detection rate and signal transmittance.",
                "parameters": [
                    {"name": "extra_loss_db", "label": "Extra Loss (dB)", "min": 1.0, "max": 15.0, "default": 5.0}
                ],
            },
            {
                "id": "timing",
                "name": "Timing Jitter / Gate Displacement",
                "target": "Arrival Time Synchronization",
                "mechanism": "Eve delays or shifts pulse arrival times relative to Bob's detection gate window.",
                "default_severity": 0.7,
                "expected_signature": "Timing offset variance and gate edge photon loss.",
                "parameters": [
                    {"name": "jitter_std_ns", "label": "Timing Jitter (ns)", "min": 0.1, "max": 2.0, "default": 1.0}
                ],
            },
            {
                "id": "source_manipulation",
                "name": "Source Intensity Manipulation",
                "target": "Alice Laser Diode",
                "mechanism": "Eve alters source mean photon number mu via Trojan-horse side channel.",
                "default_severity": 0.6,
                "expected_signature": "Abnormal Poisson click statistics and multiphoton probability.",
                "parameters": [
                    {"name": "intensity_multiplier", "label": "Intensity Multiplier", "min": 1.1, "max": 4.0, "default": 2.0}
                ],
            },
        ]

    async def start(
        self,
        scenario_id: str = "live_default",
        model_id: str = settings.default_model_id,
        seed: int = 42,
    ) -> str:
        """Start a new live simulation session."""
        await self.stop()

        self.session_id = f"SIM-{int(time.time())}"
        self.scenario_id = scenario_id
        self.model_id = model_id
        self.window_index = 0
        self.active_attack = None

        # Build fresh QKD configuration
        cfg = QKDConfig()
        cfg.simulation.chunk_size = self.chunk_size
        cfg.telemetry.window_size = self.chunk_size

        # Initialize physical simulator and aggregator
        self._simulator = Simulator(cfg, run_id=self.session_id, seed=seed, scenario_id=scenario_id)
        self._aggregator = WindowAggregator(cfg)

        # Reset model buffers and incident state
        inference_service.reset_buffer()
        incident_service.reset_session(self.session_id)

        self.status = "running"

        # Persist session record
        save_session({
            "session_id": self.session_id,
            "mode": "live_simulation",
            "status": "running",
            "scenario_id": scenario_id,
            "model_id": model_id,
            "created_at": time.time(),
        })

        log_audit(
            event_type="SESSION_START",
            message=f"Started live simulation session {self.session_id} with model {model_id}",
            session_id=self.session_id,
            metadata={"scenario_id": scenario_id, "model_id": model_id, "seed": seed},
        )

        # Launch background simulation worker
        self._task = asyncio.create_task(self._simulation_loop())
        return self.session_id

    async def pause(self) -> None:
        """Pause the running simulation."""
        if self.status == "running":
            self.status = "paused"
            log.info(f"Simulation paused at window {self.window_index}")

    async def resume(self) -> None:
        """Resume paused simulation."""
        if self.status == "paused":
            self.status = "running"
            log.info(f"Simulation resumed at window {self.window_index}")

    async def stop(self) -> None:
        """Stop and terminate the simulation session."""
        self.status = "stopped"
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("Simulation stopped")

    async def reset(self) -> None:
        """Reset the simulator to clean initial state."""
        await self.stop()
        self.window_index = 0
        self.active_attack = None
        if self._simulator:
            self._simulator.reset_runtime_state()
            self._simulator.attack_engine.clear()
            self._simulator.attacks_enabled = False
        self.status = "idle"
        log.info("Simulation reset to clean state")

    def inject_attack(
        self,
        attack_type: str,
        severity: float = 0.8,
        duration_windows: int = 15,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dynamically inject a physical attack into the active simulation."""
        if not self._simulator or self.status not in ("running", "paused"):
            raise RuntimeError("Simulation must be running to inject an attack.")

        if attack_type not in ATTACK_MAP:
            raise ValueError(f"Unsupported attack type '{attack_type}'. Available: {list(ATTACK_MAP.keys())}")

        atk_type_enum, atk_cls = ATTACK_MAP[attack_type]
        params = parameters or {}
        if attack_type == "intercept_resend" and "intensity" not in params:
            params["intensity"] = severity
        elif attack_type == "pns" and "split_ratio" not in params:
            params["split_ratio"] = severity
        elif attack_type == "channel_loss" and "extra_loss_db" not in params:
            params["extra_loss_db"] = severity * 10.0

        start_p = self.window_index * self.chunk_size
        end_p = (self.window_index + duration_windows) * self.chunk_size

        scenario = AttackScenario(
            scenario_id=f"live_{attack_type}_{self.window_index}",
            attack_type=atk_type_enum,
            severity=severity,
            start_pulse=start_p,
            end_pulse=end_p,
            parameters=params,
            description=f"Live operator injected {attack_type} at window {self.window_index}",
        )

        attack_instance = atk_cls(scenario)
        self._simulator.attack_engine.add_attack(attack_instance)
        self._simulator.attacks_enabled = True

        self.active_attack = {
            "attack_type": attack_type,
            "severity": severity,
            "start_window": self.window_index,
            "end_window": self.window_index + duration_windows,
            "parameters": params,
        }

        log_audit(
            event_type="ATTACK_INJECTED",
            message=f"Operator injected {attack_type} (severity={severity}) from window {self.window_index} to {self.window_index + duration_windows}",
            session_id=self.session_id,
            metadata=self.active_attack,
        )

        log.info(f"Injected attack {attack_type} at window {self.window_index}")
        return self.active_attack

    def clear_attack(self) -> None:
        """Clear all active attacks from the simulator."""
        if self._simulator:
            self._simulator.attack_engine.clear()
            self._simulator.attacks_enabled = False
        prev = self.active_attack
        self.active_attack = None

        log_audit(
            event_type="ATTACK_CLEARED",
            message="Operator cleared active attacks",
            session_id=self.session_id,
            metadata={"previous_attack": prev},
        )
        log.info("Cleared all active attacks")

    async def _simulation_loop(self) -> None:
        """Continuous chunked event generation and inference loop."""
        log.info(f"Starting simulation worker loop for session {self.session_id}")

        try:
            while self.status in ("running", "paused"):
                if self.status == "paused":
                    await asyncio.sleep(0.1)
                    continue

                t0 = time.time()
                chunk_idx = self.window_index
                start_pulse = chunk_idx * self.chunk_size

                # 1. Run physical chunk simulation (Alice -> Channel -> Eve -> Bob)
                events_df = self._simulator.simulate_chunk(
                    chunk_index=chunk_idx,
                    n_pulses=self.chunk_size,
                    start_pulse=start_pulse,
                )

                # 2. Extract aggregated telemetry window (130 features)
                features_dict = self._aggregator.aggregate(events_df)

                # Extract observable metrics
                qber = float(features_dict.get("qber", 0.0))
                sifted_rate = float(features_dict.get("sifted_rate", 0.0))
                sig_gain = float(features_dict.get("gain_signal", 0.0))
                dec_gain = float(features_dict.get("gain_decoy", 0.0))
                det_imbalance = float(features_dict.get("detector_efficiency_imbalance", 0.0))

                # Extract simulator ground truth (honest separation from detector prediction!)
                gt_active = bool(features_dict.get("ground_truth_attack_active", False))
                gt_type = str(features_dict.get("ground_truth_attack_type", "none"))

                # Check if an injected attack expired
                if self.active_attack and self.window_index >= self.active_attack.get("end_window", 0):
                    self.clear_attack()

                # 3. Perform genuine ML/QML model inference
                inf_result = inference_service.score_window(features_dict, model_id=self.model_id)

                # 4. Evaluate alert policy & persistent incident manager
                sim_time_ms = float(self.window_index * 100.0)
                incident = incident_service.evaluate_window(
                    session_id=self.session_id,
                    window_index=self.window_index,
                    simulated_time_ms=sim_time_ms,
                    inference_result=inf_result,
                    telemetry_point={
                        "ground_truth_attack_active": gt_active,
                        "ground_truth_attack_type": gt_type,
                        "qber": qber,
                        "sifted_rate": sifted_rate,
                        "signal_gain": sig_gain,
                        "decoy_gain": dec_gain,
                        "detector_imbalance": det_imbalance,
                        "features": features_dict,
                    },
                )

                # 5. Build streaming telemetry packet
                packet = {
                    "type": "telemetry_update",
                    "mode": "live_simulation",
                    "session_id": self.session_id,
                    "window_index": self.window_index,
                    "timestamp": time.time(),
                    "simulated_time_ms": sim_time_ms,
                    "qber": qber,
                    "sifted_rate": sifted_rate,
                    "signal_gain": sig_gain,
                    "decoy_gain": dec_gain,
                    "detector_imbalance": det_imbalance,
                    "anomaly_score": inf_result["anomaly_score"],
                    "threshold": inf_result["threshold"],
                    "is_anomaly": inf_result["is_anomaly"],
                    "inference_latency_ms": inf_result["latency_ms"],
                    "model_id": self.model_id,
                    "model_components": inf_result.get("components", {}),
                    "ground_truth_attack_active": gt_active,
                    "ground_truth_attack_type": gt_type,
                    "active_attack_operator": self.active_attack,
                    "incident": incident,
                    "display_badge": "SIMULATION ENVIRONMENT",
                    "scenario_id": self.scenario_id,
                }

                # 6. Persist telemetry point to SQLite
                save_telemetry_point({
                    "session_id": self.session_id,
                    "window_index": self.window_index,
                    "timestamp": packet["timestamp"],
                    "qber": qber,
                    "sifted_rate": sifted_rate,
                    "signal_gain": sig_gain,
                    "decoy_gain": dec_gain,
                    "detector_imbalance": det_imbalance,
                    "anomaly_score": inf_result["anomaly_score"],
                    "threshold": inf_result["threshold"],
                    "is_anomaly": inf_result["is_anomaly"],
                    "ground_truth_attack_active": gt_active,
                    "ground_truth_attack_type": gt_type,
                    "features": {k: v for k, v in features_dict.items() if isinstance(v, (int, float, bool))},
                })

                # 7. Notify WebSocket subscribers
                for sub in self._subscribers:
                    try:
                        res = sub(packet)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as e:
                        log.error(f"Subscriber notification error: {e}")

                self.window_index += 1

                # Pace the simulation loop for stable, responsive 2 Hz visual updates
                elapsed = time.time() - t0
                target_interval = settings.window_update_interval_s
                sleep_time = max(0.01, target_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            log.info("Simulation task cancelled")
        except Exception as e:
            log.error(f"Simulator worker exception: {e}", exc_info=True)
            self.status = "error"


simulator_service = SimulatorService()
