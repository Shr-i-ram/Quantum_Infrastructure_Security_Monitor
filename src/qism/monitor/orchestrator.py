"""Central monitoring orchestrator for QISM V0.9.

Coordinates the complete end-to-end operational pipeline:
    Telemetry Ingestion -> Validation -> Temporal State -> Anomaly Detection
    -> Drift Detection -> Safe Adaptation -> Signal Fusion -> Alarm Correlation
    -> Incident Lifecycle -> Severity Engine -> Explainable Alerts -> Audit Trail
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from qism.drift.adaptation import (
    AdaptationConfig,
    AdaptationDecision,
    AdaptationPolicy,
    AdaptationState,
    AdaptationStateMachine,
)
from qism.drift.detectors import (
    DriftDetectorConfig,
    DriftMetric,
    StatisticalDriftDetector,
)
from qism.drift.reference import ReferenceManager, ReferenceStrategy
from qism.drift.registry import ModelRegistry
from qism.ml.models.base import AnomalyDetector
from qism.temporal.models import RollingZScoreDetector
from qism.ml.preprocessing.scalers import PreprocessorPipeline
from qism.monitor.alerts import (
    AuditLogger,
    ConsoleNotifier,
    NotificationBackend,
)
from qism.monitor.fusion import SignalFusionConfig, SignalFusionEngine
from qism.monitor.incident import IncidentConfig, IncidentManager
from qism.monitor.ingestion import IngestionConfig, TelemetryIngestor
from qism.monitor.severity import SeverityEngine, SeverityThresholdConfig
from qism.monitor.types import (
    AlarmRecord,
    Alert,
    AuditEventType,
    DetectorSignal,
    Incident,
    IncidentState,
    MonitorRunResult,
    MonitorStepResult,
    SeverityLevel,
    TelemetryRecord,
)
from qism.temporal.alarm import AlarmConfig, OperationalState, TemporalAlarmStateMachine


@dataclass
class MonitorConfig:
    """Master configuration for the integrated QISM V0.9 monitor."""
    run_id: str = "run_v09_001"
    session_id: str = "session_001"
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    fusion: SignalFusionConfig = field(default_factory=SignalFusionConfig)
    incident: IncidentConfig = field(default_factory=IncidentConfig)
    severity: SeverityThresholdConfig = field(default_factory=SeverityThresholdConfig)
    adaptation_policy: AdaptationPolicy = AdaptationPolicy.SAFE
    history_window_length: int = 5
    drift_evaluation_window: int = 20
    anomaly_threshold: float = 2.50
    drift_threshold: float = 0.40
    enable_adaptation: bool = True
    audit_log_file: str | None = None


def make_nominal_baseline_matrix(feature_names: list[str], n_samples: int = 50, seed: int = 42) -> np.ndarray:
    """Synthesize unperturbed calibration baseline telemetry for initial detector reference."""
    rng = np.random.RandomState(seed)
    base_values = {
        "qber": (0.015, 0.002),
        "qber_z": (0.015, 0.002),
        "qber_x": (0.015, 0.002),
        "sifted_rate": (850.0, 20.0),
        "signal_gain": (0.045, 0.003),
        "decoy_gain": (0.012, 0.001),
        "vacuum_gain": (0.0008, 0.0001),
        "detector_imbalance": (0.002, 0.0005),
        "double_click_rate": (0.0004, 0.0001),
        "dark_count_rate": (25.0, 1.5),
        "sensor_temperature_mean": (21.0, 0.2),
        "sensor_temperature_std": (0.15, 0.02),
        "sensor_background_light_mean": (12.0, 0.8),
        "sensor_vibration_mean": (0.012, 0.002),
        "clock_drift_rate": (0.05, 0.01),
        "autocorr_lag1": (0.02, 0.005),
    }
    arr = np.zeros((n_samples, len(feature_names)), dtype=np.float64)
    for j, fn in enumerate(feature_names):
        mean_val, std_val = base_values.get(fn, (1.0, 0.1))
        arr[:, j] = rng.normal(mean_val, std_val, size=n_samples)
    return np.maximum(0.0, arr)


class QISMMonitor:
    """Integrated operational QKD security monitor orchestrator."""

    def __init__(
        self,
        config: MonitorConfig | None = None,
        feature_names: list[str] | None = None,
        anomaly_model: AnomalyDetector | None = None,
        notifiers: list[NotificationBackend] | None = None,
    ):
        self.config = config or MonitorConfig()
        self.feature_names = feature_names or [
            "qber", "qber_z", "qber_x", "sifted_rate", "signal_gain", "decoy_gain",
            "vacuum_gain", "detector_imbalance", "double_click_rate", "dark_count_rate",
            "sensor_temperature_mean", "sensor_temperature_std", "sensor_background_light_mean",
            "sensor_vibration_mean", "clock_drift_rate", "autocorr_lag1",
        ]

        init_baseline = make_nominal_baseline_matrix(self.feature_names)

        # 1. Ingestion Layer
        self.ingestor = TelemetryIngestor(self.config.ingestion)

        # 2. Preprocessor & Anomaly Detector
        self.preprocessor = PreprocessorPipeline(scaler="robust")
        if anomaly_model is not None:
            self.anomaly_model = anomaly_model
        else:
            det = RollingZScoreDetector(window_length=self.config.history_window_length)
            det.fit(init_baseline)
            det.threshold = self.config.anomaly_threshold
            self.anomaly_model = det

        # 3. Alarm State Machine (V0.7)
        self.alarm_machine = TemporalAlarmStateMachine(
            config=AlarmConfig(
                tau_high=self.config.anomaly_threshold,
                tau_low=self.config.anomaly_threshold * 0.70,
                persistence_k=2,
                persistence_n=4,
                cooldown_windows=3,
            )
        )

        # 4. Drift Detector & Reference Manager (V0.8)
        self.drift_detector = StatisticalDriftDetector(
            config=DriftDetectorConfig(
                metric=DriftMetric.WASSERSTEIN,
                threshold=self.config.drift_threshold,
                feature_names=self.feature_names,
            )
        )
        self.reference_manager = ReferenceManager(
            strategy=ReferenceStrategy.SLIDING,
            window_size=40,
            safe_anomaly_threshold=self.config.anomaly_threshold * 0.85,
            initial_data=init_baseline,
            feature_names=self.feature_names,
        )

        # 5. Model Registry & Safe Adaptation Machine (V0.8)
        self.registry = ModelRegistry()
        self.registry.register_initial_model(
            model=self.anomaly_model,
            threshold=self.config.anomaly_threshold,
            reference_version_id="ref_v001",
        )
        self.active_model_version = "model_v001"
        self.reference_version = "ref_v001"

        def _model_builder(cand_data: np.ndarray) -> RollingZScoreDetector:
            new_m = RollingZScoreDetector(window_length=self.config.history_window_length)
            new_m.fit(cand_data)
            new_m.threshold = self.config.anomaly_threshold
            return new_m

        self.adaptation_machine = AdaptationStateMachine(
            config=AdaptationConfig(
                policy=self.config.adaptation_policy,
                persistence_k=2,
                persistence_n=4,
                candidate_buffer_size=15,
                validation_window=10,
            ),
            registry=self.registry,
            reference_manager=self.reference_manager,
            alarm_threshold=self.config.anomaly_threshold,
            model_builder=_model_builder,
        )

        # 6. Signal Fusion Engine (V0.9)
        self.fusion_engine = SignalFusionEngine(self.config.fusion)

        # 7. Severity Engine (V0.9)
        self.severity_engine = SeverityEngine(self.config.severity)

        # 8. Incident Manager (V0.9)
        self.incident_manager = IncidentManager(
            config=self.config.incident,
            severity_engine=self.severity_engine,
        )

        # 9. Audit Logger & Alert Dispatchers (V0.9)
        self.audit_logger = AuditLogger(log_filepath=self.config.audit_log_file)
        self.notifiers = notifiers or []

        # Internal causal temporal buffers
        self._history_buffer: deque[dict[str, float]] = deque(maxlen=self.config.history_window_length)
        self._drift_eval_buffer: deque[dict[str, float]] = deque(maxlen=self.config.drift_evaluation_window)
        self._step_counter = 0

    def update(self, raw_record: TelemetryRecord | dict[str, Any]) -> MonitorStepResult:
        """Process a single sequential telemetry observation.

        Strictly causal: does not inspect any future records.
        Returns a rich MonitorStepResult.
        """
        start_t = time.perf_counter()

        # Step 1: Ingestion & Schema Validation
        try:
            record = self.ingestor.validate_and_sanitize(raw_record)
            telemetry_valid = True
        except Exception as e:
            # Safely handle validation error
            telemetry_valid = False
            self.audit_logger.log_event(
                event_type=AuditEventType.TELEMETRY_RECEIVED,
                component="ingestion",
                timestamp_s=getattr(raw_record, "timestamp_s", 0.0) if hasattr(raw_record, "timestamp_s") else 0.0,
                run_id=self.config.run_id,
                session_id=self.config.session_id,
                severity=SeverityLevel.HIGH,
                model_version=self.active_model_version,
                reference_version=self.reference_version,
                message=f"Telemetry validation rejected record: {e}",
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return MonitorStepResult(
                step_index=self._step_counter,
                timestamp_s=getattr(raw_record, "timestamp_s", 0.0) if hasattr(raw_record, "timestamp_s") else 0.0,
                telemetry_valid=False,
                anomaly_score=0.0,
                anomaly_threshold=self.config.anomaly_threshold,
                is_anomaly=False,
                drift_score=0.0,
                drift_threshold=self.config.drift_threshold,
                is_drift=False,
                evidence_score=0.0,
                active_alarms=[],
                active_incidents=[],
                new_alert=None,
                current_severity=SeverityLevel.INFO,
                adaptation_state="INVALID_TELEMETRY",
                model_version=self.active_model_version,
                reference_version=self.reference_version,
                processing_latency_ms=elapsed_ms,
            )

        self._step_counter += 1
        ts = record.timestamp_s
        feats = record.features

        # Step 2: Temporal Buffering
        self._history_buffer.append(feats)
        self._drift_eval_buffer.append(feats)

        # Step 3: Anomaly Detection
        # Feature array extraction in stable order
        feat_vec = np.array([feats.get(fn, 0.0) for fn in self.feature_names], dtype=np.float64).reshape(1, -1)

        # Compute anomaly score using active model
        if hasattr(self.anomaly_model, "score_samples"):
            raw_scores = self.anomaly_model.score_samples(feat_vec)
            anomaly_score = float(raw_scores[0])
        elif hasattr(self.anomaly_model, "score"):
            raw_scores = self.anomaly_model.score(feat_vec)
            anomaly_score = float(raw_scores[0])
        elif hasattr(self.anomaly_model, "decision_function"):
            anomaly_score = float(-self.anomaly_model.decision_function(feat_vec)[0])
        else:
            # Fallback simple deviation norm
            anomaly_score = float(np.linalg.norm(feat_vec))

        anom_thresh = self.config.anomaly_threshold
        is_anom = anomaly_score >= anom_thresh

        # Step 4: Temporal Alarm State Machine
        alarm_step = self.alarm_machine.step(anomaly_score, window_idx=self._step_counter, timestamp_s=ts)
        alarms: list[AlarmRecord] = []
        if alarm_step.raw_alarm or alarm_step.filtered_alarm:
            sev_alarm = SeverityLevel.HIGH if alarm_step.filtered_alarm else SeverityLevel.MEDIUM
            alm = AlarmRecord(
                alarm_id=f"ALM-{self._step_counter:05d}",
                source="temporal_alarm_machine",
                timestamp_s=ts,
                severity=sev_alarm,
                is_active=alarm_step.filtered_alarm,
                persistence_count=alarm_step.consecutive_high_count,
                message=f"Temporal alarm state: {alarm_step.state.value} (score={anomaly_score:.3f})",
                signals=["temporal_anomaly_model"],
            )
            alarms.append(alm)

        # Step 5: Drift Detection
        drift_score = 0.0
        drift_thresh = self.config.drift_threshold
        is_drift = False

        ref_samples = self.reference_manager.get_reference_samples()
        curr_eval_samples = np.array(
            [[row.get(fn, 0.0) for fn in self.feature_names] for row in self._drift_eval_buffer],
            dtype=np.float64,
        )

        drift_report = self.drift_detector.compute_drift(
            reference_samples=ref_samples,
            current_samples=curr_eval_samples,
            step_index=self._step_counter,
            timestamp_s=ts,
        )
        drift_score = float(drift_report.drift_score)
        drift_thresh = float(drift_report.threshold)
        is_drift = bool(drift_report.is_drift)

        # Step 6: Safe Adaptation Step
        adaptation_decision = AdaptationDecision.NO_DRIFT
        adapt_state_str = "DISABLED"
        if self.config.enable_adaptation:
            adapt_res = self.adaptation_machine.step(
                sample=feat_vec[0],
                anomaly_score=anomaly_score,
                operational_alarm=bool(alarm_step.filtered_alarm),
                drift_report=drift_report,
                step_index=self._step_counter,
                timestamp_s=ts,
            )
            adaptation_decision = adapt_res.decision
            adapt_state_str = adapt_res.state.value

            if adaptation_decision == AdaptationDecision.ADAPTATION_INITIATED:
                self.audit_logger.log_event(
                    event_type=AuditEventType.ADAPTATION_STARTED,
                    component="adaptation_machine",
                    timestamp_s=ts,
                    run_id=self.config.run_id,
                    session_id=self.config.session_id,
                    anomaly_score=anomaly_score,
                    drift_score=drift_score,
                    model_version=self.active_model_version,
                    reference_version=self.reference_version,
                    message="Initiated safe candidate adaptation gathering.",
                )
            elif adaptation_decision == AdaptationDecision.PROMOTED:
                self.active_model_version = self.registry.active_version_id
                self.audit_logger.log_event(
                    event_type=AuditEventType.MODEL_PROMOTED,
                    component="adaptation_machine",
                    timestamp_s=ts,
                    run_id=self.config.run_id,
                    session_id=self.config.session_id,
                    model_version=self.active_model_version,
                    reference_version=self.reference_version,
                    message=f"Candidate model successfully validated and promoted to {self.active_model_version}.",
                )
            elif adaptation_decision in (
                AdaptationDecision.REJECTED_ANOMALY,
                AdaptationDecision.REJECTED_VALIDATION_FAILED,
            ):
                self.audit_logger.log_event(
                    event_type=AuditEventType.ADAPTATION_REJECTED,
                    component="adaptation_machine",
                    timestamp_s=ts,
                    run_id=self.config.run_id,
                    session_id=self.config.session_id,
                    anomaly_score=anomaly_score,
                    drift_score=drift_score,
                    model_version=self.active_model_version,
                    reference_version=self.reference_version,
                    message="Candidate adaptation rejected due to validation failure or anomaly contamination.",
                )


        # Step 7: Signal Fusion
        feat_contribs = {}
        if is_anom:
            # Approximate relative deviation contributions
            feat_contribs = {fn: float(abs(feats.get(fn, 0.0))) for fn in self.feature_names[:5]}

        evidence_score, signals, fusion_breakdown = self.fusion_engine.evaluate(
            anomaly_score=anomaly_score,
            anomaly_threshold=anom_thresh,
            drift_score=drift_score,
            drift_threshold=drift_thresh,
            features=feats,
            timestamp_s=ts,
            feature_contributions=feat_contribs,
        )

        # Step 8: Incident Correlation & Lifecycle
        affected_feats = [s["name"] for s in fusion_breakdown["dominant_signals"] if s["norm_val"] >= 0.40]
        active_inc, new_alert = self.incident_manager.process_step(
            timestamp_s=ts,
            evidence_score=evidence_score,
            signals=signals,
            alarms=alarms,
            affected_features=affected_feats,
            active_model_version=self.active_model_version,
            active_reference_version=self.reference_version,
            sequence_id=record.sequence_id,
            drift_active=is_drift,
        )

        current_sev = active_inc.severity if active_inc else SeverityLevel.INFO

        # Step 9: Audit Logging & Dispatching
        if is_anom:
            self.audit_logger.log_event(
                event_type=AuditEventType.ANOMALY_DETECTED,
                component="anomaly_engine",
                timestamp_s=ts,
                run_id=self.config.run_id,
                session_id=self.config.session_id,
                anomaly_score=anomaly_score,
                evidence_score=evidence_score,
                model_version=self.active_model_version,
                reference_version=self.reference_version,
                message=f"Anomaly score {anomaly_score:.3f} >= threshold {anom_thresh:.3f}",
            )

        if is_drift:
            self.audit_logger.log_event(
                event_type=AuditEventType.DRIFT_DETECTED,
                component="drift_engine",
                timestamp_s=ts,
                run_id=self.config.run_id,
                session_id=self.config.session_id,
                drift_score=drift_score,
                evidence_score=evidence_score,
                model_version=self.active_model_version,
                reference_version=self.reference_version,
                message=f"Drift score {drift_score:.3f} >= threshold {drift_thresh:.3f}",
            )

        if new_alert is not None:
            # Log alert
            evt_type = AuditEventType.INCIDENT_OPENED if new_alert.incident_state == IncidentState.OPENING else (
                AuditEventType.INCIDENT_RESOLVED if new_alert.incident_state == IncidentState.RESOLVED else AuditEventType.INCIDENT_UPDATED
            )
            self.audit_logger.log_event(
                event_type=evt_type,
                component="incident_manager",
                timestamp_s=ts,
                run_id=self.config.run_id,
                session_id=self.config.session_id,
                incident_id=new_alert.incident_id,
                alert_id=new_alert.alert_id,
                severity=new_alert.severity,
                anomaly_score=anomaly_score,
                drift_score=drift_score,
                evidence_score=evidence_score,
                model_version=self.active_model_version,
                reference_version=self.reference_version,
                message=new_alert.explanation_summary,
            )

            # Dispatch alert to registered channels
            for notifier in self.notifiers:
                try:
                    notifier.dispatch(new_alert)
                except Exception:
                    pass

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return MonitorStepResult(
            step_index=self._step_counter,
            timestamp_s=ts,
            telemetry_valid=True,
            anomaly_score=anomaly_score,
            anomaly_threshold=anom_thresh,
            is_anomaly=is_anom,
            drift_score=drift_score,
            drift_threshold=drift_thresh,
            is_drift=is_drift,
            evidence_score=evidence_score,
            active_alarms=alarms,
            active_incidents=[active_inc] if active_inc else [],
            new_alert=new_alert,
            current_severity=current_sev,
            adaptation_state=adapt_state_str,
            model_version=self.active_model_version,
            reference_version=self.reference_version,
            processing_latency_ms=elapsed_ms,
            signals=signals,
        )

    def run(
        self,
        stream: Iterable[TelemetryRecord] | pd.DataFrame,
        feature_cols: Sequence[str] | None = None,
    ) -> MonitorRunResult:
        """Run batch or stream evaluation sequentially without future lookahead."""
        start_t = time.perf_counter()
        step_results: list[MonitorStepResult] = []

        if isinstance(stream, pd.DataFrame):
            record_iter = self.ingestor.stream_from_dataframe(
                stream, feature_cols=feature_cols, run_id=self.config.run_id
            )
        else:
            record_iter = stream

        for rec in record_iter:
            res = self.update(rec)
            step_results.append(res)

        # Finalize open incidents
        last_ts = step_results[-1].timestamp_s if step_results else 0.0
        all_incidents = self.incident_manager.finalize(final_timestamp_s=last_ts)
        all_alerts = list(self.incident_manager.all_emitted_alerts)

        total_duration = time.perf_counter() - start_t
        raw_alarms = self.incident_manager.total_raw_alarms
        comp_ratio = self.incident_manager.compression_ratio

        latencies = [r.processing_latency_ms for r in step_results]

        metrics = {
            "total_steps": len(step_results),
            "total_incidents": len(all_incidents),
            "total_alerts": len(all_alerts),
            "total_raw_alarms": raw_alarms,
            "compression_ratio": float(comp_ratio),
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "max_latency_ms": float(np.max(latencies)) if latencies else 0.0,
            "total_runtime_s": float(total_duration),
            "throughput_steps_per_s": float(len(step_results) / max(1e-4, total_duration)),
        }

        return MonitorRunResult(
            total_steps=len(step_results),
            total_duration_s=total_duration,
            step_results=step_results,
            all_incidents=all_incidents,
            all_alerts=all_alerts,
            audit_events=list(self.audit_logger.events),
            metrics=metrics,
        )

    def checkpoint(self, checkpoint_path: str | Path) -> None:
        """Serialize complete monitor state for streaming recovery."""
        p = Path(checkpoint_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "step_counter": self._step_counter,
            "active_model_version": self.active_model_version,
            "reference_version": self.reference_version,
            "alarm_state": self.alarm_machine.current_state.value,
            "alarm_duration": self.alarm_machine.state_duration,
            "adaptation_state": self.adaptation_machine.state.value,
            "adaptation_policy": self.adaptation_machine.config.policy.value,
            "total_raw_alarms": self.incident_manager.total_raw_alarms,
            "active_incident": self.incident_manager.active_incident.to_dict() if self.incident_manager.active_incident else None,
            "resolved_incidents": [inc.to_dict() for inc in self.incident_manager.resolved_incidents],
            "history_buffer": list(self._history_buffer),
            "audit_event_count": len(self.audit_logger.events),
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def restore(self, checkpoint_path: str | Path) -> None:
        """Restore complete monitor state from a saved checkpoint."""
        p = Path(checkpoint_path)
        with open(p, "r", encoding="utf-8") as f:
            state = json.load(f)

        self._step_counter = state["step_counter"]
        self.active_model_version = state["active_model_version"]
        self.reference_version = state["reference_version"]
        self.alarm_machine.current_state = OperationalState(state["alarm_state"])
        self.alarm_machine.state_duration = state.get("alarm_duration", 0)
        self.adaptation_machine.state = AdaptationState(state["adaptation_state"])
        self.incident_manager.total_raw_alarms = state["total_raw_alarms"]

        # Restore history buffer
        self._history_buffer.clear()
        for h in state.get("history_buffer", []):
            self._history_buffer.append(h)

        # Restore active incident
        if state.get("active_incident"):
            d = state["active_incident"]
            self.incident_manager.active_incident = Incident(
                incident_id=d["incident_id"],
                start_time_s=d["start_time_s"],
                end_time_s=d.get("end_time_s"),
                duration_s=d["duration_s"],
                state=IncidentState(d["state"]),
                severity=SeverityLevel(d["severity"]),
                peak_severity=SeverityLevel(d["peak_severity"]),
                evidence_score=d["evidence_score"],
                peak_evidence_score=d["peak_evidence_score"],
                constituent_alarms=[],
                contributing_signals=[],
                affected_features=d["affected_features"],
                subsystems_involved=d["subsystems_involved"],
                explanation=d["explanation"],
                recommendation=None,
                active_model_version=d["active_model_version"],
                active_reference_version=d["active_reference_version"],
                alarm_count=d["alarm_count"],
            )
        else:
            self.incident_manager.active_incident = None

        # Restore resolved incidents
        self.incident_manager.resolved_incidents.clear()
        for d in state.get("resolved_incidents", []):
            self.incident_manager.resolved_incidents.append(
                Incident(
                    incident_id=d["incident_id"],
                    start_time_s=d["start_time_s"],
                    end_time_s=d.get("end_time_s"),
                    duration_s=d["duration_s"],
                    state=IncidentState(d["state"]),
                    severity=SeverityLevel(d["severity"]),
                    peak_severity=SeverityLevel(d["peak_severity"]),
                    evidence_score=d["evidence_score"],
                    peak_evidence_score=d["peak_evidence_score"],
                    constituent_alarms=[],
                    contributing_signals=[],
                    affected_features=d.get("affected_features", []),
                    subsystems_involved=d.get("subsystems_involved", []),
                    explanation=d.get("explanation", ""),
                    recommendation=None,
                    active_model_version=d.get("active_model_version", self.active_model_version),
                    active_reference_version=d.get("active_reference_version", self.reference_version),
                    alarm_count=d.get("alarm_count", 1),
                )
            )

    def reset(self) -> None:
        """Reset all orchestrator components."""
        self.ingestor.reset()
        self.fusion_engine.reset()
        self.incident_manager.reset()
        self.audit_logger.reset()
        self.alarm_machine.reset()
        self.adaptation_machine.reset()
        self._history_buffer.clear()
        self._drift_eval_buffer.clear()
        self._step_counter = 0
