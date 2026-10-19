"""Concept Drift, Distribution Shift, and Safe Model Adaptation Package for QISM V0.8."""

from qism.drift.detectors import (
    AggregationStrategy,
    DriftDetectorConfig,
    DriftMetric,
    DriftReport,
    StatisticalDriftDetector,
    compute_ks_distance,
    compute_psi_distance,
    compute_wasserstein_distance,
)
from qism.drift.reference import (
    ReferenceManager,
    ReferenceMetadata,
    ReferenceStrategy,
)
from qism.drift.registry import (
    AdaptationDecision,
    AdaptationLogEntry,
    ModelRegistry,
    ModelStatus,
    ModelVersionRecord,
)
from qism.drift.adaptation import (
    AdaptationConfig,
    AdaptationPolicy,
    AdaptationState,
    AdaptationStateMachine,
    AdaptationStepResult,
)
from qism.drift.scenarios import (
    DriftScenarioGenerator,
    DriftScenarioType,
    ScenarioSpec,
)
from qism.drift.streaming import (
    AdaptiveStreamingEngine,
    AdaptiveStreamingResult,
)
from qism.drift.evaluation import (
    DriftBenchmarkMetrics,
    DriftEvaluator,
)
from qism.drift.plots import (
    plot_distribution_comparison,
    plot_drift_heatmap,
    plot_drift_telemetry_timeline,
    plot_fixed_vs_naive_vs_safe,
)
from qism.drift.experiments import (
    DriftExperimentRunner,
    DriftExperimentSuiteConfig,
)

__all__ = [
    "AggregationStrategy",
    "DriftDetectorConfig",
    "DriftMetric",
    "DriftReport",
    "StatisticalDriftDetector",
    "compute_ks_distance",
    "compute_psi_distance",
    "compute_wasserstein_distance",
    "ReferenceManager",
    "ReferenceMetadata",
    "ReferenceStrategy",
    "AdaptationDecision",
    "AdaptationLogEntry",
    "ModelRegistry",
    "ModelStatus",
    "ModelVersionRecord",
    "AdaptationConfig",
    "AdaptationPolicy",
    "AdaptationState",
    "AdaptationStateMachine",
    "AdaptationStepResult",
    "DriftScenarioGenerator",
    "DriftScenarioType",
    "ScenarioSpec",
    "AdaptiveStreamingEngine",
    "AdaptiveStreamingResult",
    "DriftBenchmarkMetrics",
    "DriftEvaluator",
    "plot_distribution_comparison",
    "plot_drift_heatmap",
    "plot_drift_telemetry_timeline",
    "plot_fixed_vs_naive_vs_safe",
    "DriftExperimentRunner",
    "DriftExperimentSuiteConfig",
]
