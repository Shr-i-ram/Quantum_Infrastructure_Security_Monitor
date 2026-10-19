"""QISM V0.9 Integrated Security Monitor & Incident Analysis Package.

Public API exports for telemetry ingestion, temporal state, signal fusion,
alarm correlation, incident lifecycle, explainable severity, structured alerting,
and audit trails.
"""

from qism.monitor.alerts import (
    AuditLogger,
    ConsoleNotifier,
    FileNotifier,
    JSONNotifier,
    NotificationBackend,
)
from qism.monitor.evaluation import (
    EvaluationReport,
    GroundTruthEvent,
    evaluate_monitor_run,
    extract_ground_truth_events,
)
from qism.monitor.fusion import (
    FusionMethod,
    SignalFusionConfig,
    SignalFusionEngine,
    SignalWeightConfig,
)
from qism.monitor.incident import (
    IncidentConfig,
    IncidentManager,
)
from qism.monitor.ingestion import (
    DEFAULT_PHYSICAL_BOUNDS,
    FORBIDDEN_GROUND_TRUTH_KEYS,
    IngestionConfig,
    TelemetryIngestor,
    TelemetryValidationError,
)
from qism.monitor.orchestrator import (
    MonitorConfig,
    QISMMonitor,
)
from qism.monitor.severity import (
    SeverityEngine,
    SeverityThresholdConfig,
)
from qism.monitor.types import (
    AlarmRecord,
    Alert,
    AuditEvent,
    AuditEventType,
    DetectorSignal,
    Incident,
    IncidentState,
    MonitorRunResult,
    MonitorStepResult,
    ResponseRecommendation,
    SeverityLevel,
    TelemetryRecord,
)

__all__ = [
    # Types & Dataclasses
    "IncidentState",
    "SeverityLevel",
    "AuditEventType",
    "FusionMethod",
    "TelemetryRecord",
    "DetectorSignal",
    "AlarmRecord",
    "ResponseRecommendation",
    "Incident",
    "Alert",
    "AuditEvent",
    "MonitorStepResult",
    "MonitorRunResult",
    # Ingestion
    "TelemetryIngestor",
    "IngestionConfig",
    "TelemetryValidationError",
    "FORBIDDEN_GROUND_TRUTH_KEYS",
    "DEFAULT_PHYSICAL_BOUNDS",
    # Fusion
    "SignalFusionEngine",
    "SignalFusionConfig",
    "SignalWeightConfig",
    # Incident
    "IncidentManager",
    "IncidentConfig",
    # Severity
    "SeverityEngine",
    "SeverityThresholdConfig",
    # Alerts & Audit
    "NotificationBackend",
    "ConsoleNotifier",
    "FileNotifier",
    "JSONNotifier",
    "AuditLogger",
    # Orchestrator
    "QISMMonitor",
    "MonitorConfig",
    # Evaluation
    "evaluate_monitor_run",
    "EvaluationReport",
    "GroundTruthEvent",
    "extract_ground_truth_events",
    # V1.0 Benchmarks & Demonstration
    "run_v10_benchmarks",
    "run_v10_demo",
]

from qism.monitor.v10_benchmark import run_v10_benchmarks, run_v10_demo
