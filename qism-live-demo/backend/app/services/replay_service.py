"""Recorded Scenario Replay Service for QISM Live Demonstration Console.

Streams verified, research-grade telemetry recorded during the Phase X benchmark.
Feeds genuine telemetry windows into the active inference pipeline with explicit
provenance and "RECORDED SCENARIO REPLAY" labeling.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import time
from typing import Any, Callable

import pandas as pd

from ..config import DATASET_ROOT
from ..database import log_audit, save_session, save_telemetry_point
from .incident_service import incident_service
from .inference_service import inference_service

log = logging.getLogger("qism.demo.replay")


class ReplayService:
    """Manages playback of pre-recorded QKD telemetry datasets with live inference."""

    def __init__(self) -> None:
        self.status: str = "idle"  # idle, replaying, paused, stopped
        self.session_id: str | None = None
        self.scenario_id: str | None = None
        self.model_id: str = "PITQSVDD_6Q_Angle"
        self.current_window: int = 0
        self.total_windows: int = 0
        self.speed: float = 1.0  # multiplier (1.0 = ~0.5s per window)
        self.live_inference: bool = True
        self._df: pd.DataFrame | None = None
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
    def get_catalog() -> list[dict[str, Any]]:
        """Return catalog of verified recorded scenarios with provenance."""
        catalog = [
            {
                "id": "scenario_pns_attack",
                "name": "Photon Number Splitting (PNS) Attack Scenario",
                "attack_type": "pns",
                "description": "Eve selectively splits multi-photon pulses on decoy-state channel. Displays subtle gain shift before QBER spikes.",
                "path": "data/qism_research_dataset/test_in_distribution/shard_0000/attacks/run_0000/telemetry.parquet",
                "total_windows": 20,
                "normal_windows": "0 - 4",
                "attack_windows": "5 - 19",
                "provenance": "Phase X Benchmark Dataset (test_in_distribution, shard_0000, run_0000)",
                "recommended_detector": "PITQSVDD_6Q_Angle",
            },
            {
                "id": "scenario_intercept_resend",
                "name": "Intercept-Resend Attack Scenario",
                "attack_type": "intercept_resend",
                "description": "Eve measures pulses in random bases and resends. Dramatic QBER increase (~25%) accompanied by severe state distortion.",
                "path": "data/qism_research_dataset/test_in_distribution/shard_0000/attacks/run_0003/telemetry.parquet",
                "total_windows": 20,
                "normal_windows": "0 - 4",
                "attack_windows": "5 - 19",
                "provenance": "Phase X Benchmark Dataset (test_in_distribution, shard_0000, run_0003)",
                "recommended_detector": "PITQSVDD_6Q_Angle",
            },
            {
                "id": "scenario_channel_loss",
                "name": "Channel Loss Attack Scenario",
                "attack_type": "channel_loss",
                "description": "Eve adds physical loss to mask eavesdropping. Causes precipitous drop in signal and decoy gains.",
                "path": "data/qism_research_dataset/test_in_distribution/shard_0000/attacks/run_0001/telemetry.parquet",
                "total_windows": 20,
                "normal_windows": "0 - 4",
                "attack_windows": "5 - 19",
                "provenance": "Phase X Benchmark Dataset (test_in_distribution, shard_0000, run_0001)",
                "recommended_detector": "PITQSVDD_6Q_Angle",
            },
            {
                "id": "scenario_normal_baseline",
                "name": "Normal Stable Operation Baseline",
                "attack_type": "none",
                "description": "Stable fiber link under clean environmental conditions without any attack injection. Zero false alarms.",
                "path": "data/qism_research_dataset/validation/shard_0000/baseline/run_0000/telemetry.parquet",
                "total_windows": 20,
                "normal_windows": "0 - 19",
                "attack_windows": "None",
                "provenance": "Phase X Benchmark Dataset (validation baseline, shard_0000, run_0000)",
                "recommended_detector": "PITQSVDD_6Q_Angle",
            },
        ]
        return catalog

    def load_scenario(self, scenario_id: str) -> bool:
        """Load the parquet dataframe for a scenario."""
        cat = {item["id"]: item for item in self.get_catalog()}
        entry = cat.get(scenario_id)
        if not entry:
            log.error(f"Unknown scenario_id: {scenario_id}")
            return False

        path = DATASET_ROOT.parent.parent / entry["path"]
        if not path.exists():
            # Fallback path inside DATASET_ROOT
            path = DATASET_ROOT / entry["path"].replace("data/qism_research_dataset/", "")

        if not path.exists():
            log.error(f"Parquet file not found at {path}")
            return False

        try:
            self._df = pd.read_parquet(path)
            self.total_windows = len(self._df)
            self.scenario_id = scenario_id
            self.current_window = 0
            log.info(f"Loaded scenario {scenario_id} with {self.total_windows} windows from {path}")
            return True
        except Exception as e:
            log.error(f"Failed to read parquet {path}: {e}")
            return False

    async def start(
        self,
        scenario_id: str,
        model_id: str = "PITQSVDD_6Q_Angle",
        speed: float = 1.0,
        live_inference: bool = True,
    ) -> str:
        """Start a new recorded replay session."""
        await self.stop()

        if not self.load_scenario(scenario_id):
            raise ValueError(f"Could not load scenario {scenario_id}")

        self.session_id = f"REPLAY-{int(time.time())}"
        self.model_id = model_id
        self.speed = max(0.2, min(speed, 5.0))
        self.live_inference = live_inference
        self.status = "replaying"
        self.current_window = 0

        # Reset model buffers and incident state
        inference_service.reset_buffer()
        incident_service.reset_session(self.session_id)

        # Persist session record
        save_session({
            "session_id": self.session_id,
            "mode": "recorded_replay",
            "status": "replaying",
            "scenario_id": scenario_id,
            "model_id": model_id,
            "created_at": time.time(),
        })

        log_audit(
            event_type="REPLAY_START",
            message=f"Started recorded replay {scenario_id} with model {model_id} at {self.speed}x speed",
            session_id=self.session_id,
            metadata={"scenario_id": scenario_id, "model_id": model_id, "speed": self.speed},
        )

        # Launch playback task
        self._task = asyncio.create_task(self._playback_loop())
        return self.session_id

    async def pause(self) -> None:
        """Pause playback."""
        if self.status == "replaying":
            self.status = "paused"
            log.info(f"Replay paused at window {self.current_window}")

    async def resume(self) -> None:
        """Resume playback."""
        if self.status == "paused":
            self.status = "replaying"
            log.info(f"Replay resumed from window {self.current_window}")

    async def stop(self) -> None:
        """Stop replay session."""
        self.status = "stopped"
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("Replay stopped")

    def seek(self, window_index: int) -> bool:
        """Jump to a specific window index."""
        if self._df is None or window_index < 0 or window_index >= self.total_windows:
            return False
        self.current_window = window_index
        log.info(f"Seeked replay to window {window_index}")
        return True

    def set_speed(self, speed: float) -> None:
        """Update playback speed multiplier."""
        self.speed = max(0.2, min(speed, 5.0))

    async def _playback_loop(self) -> None:
        """Background asynchronous playback worker."""
        log.info(f"Starting replay loop for session {self.session_id}")
        base_interval = 0.5  # seconds

        try:
            while self.current_window < self.total_windows:
                if self.status == "paused":
                    await asyncio.sleep(0.1)
                    continue

                if self.status != "replaying":
                    break

                t0 = time.time()
                row = self._df.iloc[self.current_window]
                features_dict = row.to_dict()

                # Extract observable metrics
                qber = float(features_dict.get("QBER", features_dict.get("qber", 0.0)))
                sifted_rate = float(features_dict.get("sifted_key_rate", features_dict.get("sifted_rate", 0.0)))
                sig_gain = float(features_dict.get("gain_signal", features_dict.get("signal_gain", 0.0)))
                dec_gain = float(features_dict.get("gain_decoy", features_dict.get("decoy_gain", 0.0)))
                det_imbalance = float(features_dict.get("detector_efficiency_imbalance", features_dict.get("detector_imbalance", 0.0)))

                gt_active = bool(features_dict.get("ground_truth_attack_active", False))
                gt_type = str(features_dict.get("ground_truth_attack_type", "none"))

                # Perform actual model inference on the telemetry window
                if self.live_inference:
                    inf_result = inference_service.score_window(features_dict, model_id=self.model_id)
                else:
                    inf_result = {
                        "model_id": self.model_id,
                        "anomaly_score": 0.0,
                        "threshold": 0.05,
                        "is_anomaly": False,
                        "latency_ms": 0.0,
                        "components": {},
                    }

                # Evaluate for incident creation
                sim_time_ms = float(self.current_window * 100.0)  # nominal 100ms per window
                incident = incident_service.evaluate_window(
                    session_id=self.session_id,
                    window_index=self.current_window,
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

                # Construct stream packet
                packet = {
                    "type": "telemetry_update",
                    "mode": "recorded_replay",
                    "session_id": self.session_id,
                    "window_index": self.current_window,
                    "total_windows": self.total_windows,
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
                    "incident": incident,
                    "display_badge": "RECORDED SCENARIO REPLAY",
                    "scenario_id": self.scenario_id,
                }

                # Persist telemetry point
                save_telemetry_point({
                    "session_id": self.session_id,
                    "window_index": self.current_window,
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

                # Broadcast to subscribers
                for sub in self._subscribers:
                    try:
                        res = sub(packet)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as e:
                        log.error(f"Error notifying subscriber: {e}")

                self.current_window += 1

                # Pace playback based on configured speed
                elapsed = time.time() - t0
                target_interval = base_interval / self.speed
                sleep_time = max(0.01, target_interval - elapsed)
                await asyncio.sleep(sleep_time)

            self.status = "completed"
            log.info(f"Replay completed for session {self.session_id}")

        except asyncio.CancelledError:
            log.info("Playback task cancelled")
        except Exception as e:
            log.error(f"Replay error: {e}", exc_info=True)
            self.status = "error"


replay_service = ReplayService()
