"""End-to-end concept drift and safe adaptation benchmark suite (V0.8).

Orchestrates Experiments 01 through 12:
- Exp 01: Stationary normal operation (baseline false alarms)
- Exp 02: Abrupt benign drift (drift detection delay)
- Exp 03: Gradual benign drift (gradual tracking and adaptation)
- Exp 04: Recurring drift (A -> B -> A recovery)
- Exp 05: Attack without drift (security baseline)
- Exp 06: Drift followed by attack (sensitivity preservation)
- Exp 07: Attack during drift (stress test)
- Exp 08: Fixed vs Naive vs Safe Adaptation (headline comparison)
- Exp 09: Drift detector comparison (Wasserstein vs KS vs PSI)
- Exp 10: Adaptation window study (short vs medium vs long)
- Exp 11: Checkpoint recovery and rollback verification
- Exp 12: Out-Of-Distribution (OOD) Normal Generalization

Serializes full manifest.json, summary.csv, and publication-grade plots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any

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
    AggregationStrategy,
    DriftDetectorConfig,
    DriftMetric,
    StatisticalDriftDetector,
)
from qism.drift.evaluation import DriftBenchmarkMetrics, DriftEvaluator
from qism.drift.plots import (
    plot_distribution_comparison,
    plot_drift_heatmap,
    plot_drift_telemetry_timeline,
    plot_fixed_vs_naive_vs_safe,
)
from qism.drift.reference import ReferenceManager, ReferenceStrategy
from qism.drift.registry import ModelRegistry
from qism.drift.scenarios import DriftScenarioGenerator, DriftScenarioType, ScenarioSpec
from qism.drift.streaming import AdaptiveStreamingEngine
from qism.ml.models.base import AnomalyDetector
from qism.ml.preprocessing.scalers import PreprocessorPipeline
from qism.temporal.alarm import AlarmConfig, TemporalAlarmStateMachine
from qism.temporal.models import RollingZScoreDetector

logger = logging.getLogger(__name__)


@dataclass
class DriftExperimentSuiteConfig:
    """Configuration for running the complete V0.8 experiment suite."""
    output_dir: str = "outputs/v08/experiments"
    master_seed: int = 42
    total_steps: int = 150
    step_duration_s: float = 0.002
    pulses_per_window: int = 2000
    alarm_threshold: float = 3.0
    drift_threshold: float = 0.45
    history_window_length: int = 5
    drift_evaluation_window: int = 20


class DriftExperimentRunner:
    """Executes the full V0.8 concept drift experimental benchmark matrix."""

    def __init__(self, config: DriftExperimentSuiteConfig | None = None):
        self.config = config or DriftExperimentSuiteConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = DriftEvaluator(
            step_duration_s=self.config.step_duration_s,
            pulses_per_window=self.config.pulses_per_window,
        )

    def _setup_engine(
        self,
        X_train_normal: np.ndarray,
        feature_names: list[str],
        policy: AdaptationPolicy = AdaptationPolicy.SAFE,
        drift_metric: DriftMetric = DriftMetric.WASSERSTEIN,
        ref_strategy: ReferenceStrategy = ReferenceStrategy.ANCHORED,
        cand_buffer_size: int = 30,
        val_window: int = 20,
    ) -> tuple[AdaptiveStreamingEngine, PreprocessorPipeline]:
        """Instantiate preprocessor, detector, drift monitor, and adaptive streaming engine."""
        preprocessor = PreprocessorPipeline(scaler="robust")
        X_train_scaled = preprocessor.fit_transform(pd.DataFrame(X_train_normal, columns=feature_names))

        detector = RollingZScoreDetector(window_length=self.config.history_window_length)
        detector.fit(X_train_scaled)
        detector.set_threshold(self.config.alarm_threshold)

        alarm_sm = TemporalAlarmStateMachine(
            AlarmConfig(tau_high=self.config.alarm_threshold, persistence_k=2, persistence_n=3, cooldown_windows=2)
        )

        drift_cfg = DriftDetectorConfig(
            metric=drift_metric,
            aggregation=AggregationStrategy.MEAN,
            threshold=self.config.drift_threshold,
            feature_names=feature_names,
        )
        drift_det = StatisticalDriftDetector(drift_cfg)

        ref_mgr = ReferenceManager(
            strategy=ref_strategy,
            window_size=50,
            safe_anomaly_threshold=self.config.alarm_threshold * 0.80,
            initial_data=X_train_scaled,
            feature_names=feature_names,
        )

        registry = ModelRegistry(output_dir=self.output_dir / "registry")
        registry.register_initial_model(
            model=detector,
            threshold=self.config.alarm_threshold,
            reference_version_id=ref_mgr.current_version_id,
        )

        def model_builder(cand_data: np.ndarray) -> AnomalyDetector:
            new_model = RollingZScoreDetector(window_length=self.config.history_window_length)
            new_model.fit(cand_data)
            new_model.set_threshold(self.config.alarm_threshold)
            return new_model

        adapt_cfg = AdaptationConfig(
            policy=policy,
            persistence_k=3,
            persistence_n=5,
            safe_anomaly_ratio=0.80,
            candidate_buffer_size=cand_buffer_size,
            validation_window=val_window,
            cooldown_steps=25,
            rollback_enabled=True,
        )
        adapt_sm = AdaptationStateMachine(
            config=adapt_cfg,
            registry=registry,
            reference_manager=ref_mgr,
            alarm_threshold=self.config.alarm_threshold,
            model_builder=model_builder,
        )

        engine = AdaptiveStreamingEngine(
            active_model=detector,
            preprocessor=preprocessor,
            alarm_machine=alarm_sm,
            drift_detector=drift_det,
            reference_manager=ref_mgr,
            adaptation_machine=adapt_sm,
            registry=registry,
            history_window_length=self.config.history_window_length,
            drift_evaluation_window=self.config.drift_evaluation_window,
            feature_names=feature_names,
        )
        return engine, preprocessor

    def run_all(self) -> dict[str, Any]:
        """Execute Experiments 1 through 12 sequentially and save artifacts."""
        all_metrics: list[DriftBenchmarkMetrics] = []
        manifest: dict[str, Any] = {
            "version": "0.8.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "master_seed": self.config.master_seed,
            "experiments": {},
        }

        # Baseline training data (stationary normal operation)
        spec_baseline = ScenarioSpec(
            scenario_type=DriftScenarioType.STATIONARY,
            total_steps=80,
            seed=self.config.master_seed,
        )
        df_train_raw, feature_cols = DriftScenarioGenerator(spec_baseline).generate_stream()
        X_train_raw = df_train_raw[feature_cols].values

        # -------------------------------------------------------------
        # EXPERIMENT 01: Stationary Normal Operation
        # -------------------------------------------------------------
        logger.info("Executing Experiment 01: Stationary Normal Operation...")
        spec01 = ScenarioSpec(scenario_type=DriftScenarioType.STATIONARY, total_steps=self.config.total_steps, seed=101)
        df01, _ = DriftScenarioGenerator(spec01).generate_stream()
        engine01, prep01 = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE)
        res01 = engine01.stream_trajectory(df01[feature_cols], timestamps_s=df01["timestamp"].values)
        m01 = self.evaluator.evaluate(res01, df01, "exp_01", "stationary", "safe")
        all_metrics.append(m01)
        manifest["experiments"]["exp_01"] = asdict(m01)

        # -------------------------------------------------------------
        # EXPERIMENT 02: Abrupt Benign Drift
        # -------------------------------------------------------------
        logger.info("Executing Experiment 02: Abrupt Benign Drift...")
        spec02 = ScenarioSpec(
            scenario_type=DriftScenarioType.ABRUPT_BENIGN,
            drift_start_step=40,
            total_steps=self.config.total_steps,
            seed=102,
        )
        df02, _ = DriftScenarioGenerator(spec02).generate_stream()
        engine02, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE)
        res02 = engine02.stream_trajectory(df02[feature_cols], timestamps_s=df02["timestamp"].values)
        m02 = self.evaluator.evaluate(res02, df02, "exp_02", "abrupt_benign", "safe")
        all_metrics.append(m02)
        manifest["experiments"]["exp_02"] = asdict(m02)

        plot_drift_telemetry_timeline(
            df02, res02, self.plots_dir / "exp02_abrupt_drift_timeline.png",
            title="Experiment 02: Abrupt Benign Drift Tracking & Safe Adaptation"
        )

        # -------------------------------------------------------------
        # EXPERIMENT 03: Gradual Benign Drift
        # -------------------------------------------------------------
        logger.info("Executing Experiment 03: Gradual Benign Drift...")
        spec03 = ScenarioSpec(
            scenario_type=DriftScenarioType.GRADUAL_BENIGN,
            drift_start_step=30,
            drift_end_step=80,
            total_steps=self.config.total_steps,
            seed=103,
        )
        df03, _ = DriftScenarioGenerator(spec03).generate_stream()
        engine03, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE)
        res03 = engine03.stream_trajectory(df03[feature_cols], timestamps_s=df03["timestamp"].values)
        m03 = self.evaluator.evaluate(res03, df03, "exp_03", "gradual_benign", "safe")
        all_metrics.append(m03)
        manifest["experiments"]["exp_03"] = asdict(m03)

        plot_drift_telemetry_timeline(
            df03, res03, self.plots_dir / "exp03_gradual_drift_timeline.png",
            title="Experiment 03: Gradual Benign Drift Evolution"
        )

        # -------------------------------------------------------------
        # EXPERIMENT 04: Recurring Drift (A -> B -> A)
        # -------------------------------------------------------------
        logger.info("Executing Experiment 04: Recurring Drift...")
        spec04 = ScenarioSpec(
            scenario_type=DriftScenarioType.RECURRING_BENIGN,
            drift_start_step=30,
            recovery_start_step=90,
            recovery_end_step=115,
            total_steps=self.config.total_steps,
            seed=104,
        )
        df04, _ = DriftScenarioGenerator(spec04).generate_stream()
        engine04, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE)
        res04 = engine04.stream_trajectory(df04[feature_cols], timestamps_s=df04["timestamp"].values)
        m04 = self.evaluator.evaluate(res04, df04, "exp_04", "recurring_benign", "safe")
        all_metrics.append(m04)
        manifest["experiments"]["exp_04"] = asdict(m04)

        plot_drift_telemetry_timeline(
            df04, res04, self.plots_dir / "exp04_recurring_drift_timeline.png",
            title="Experiment 04: Reversible Recurring Drift (A -> B -> A)"
        )

        # -------------------------------------------------------------
        # EXPERIMENT 05: Attack Without Drift
        # -------------------------------------------------------------
        logger.info("Executing Experiment 05: Attack Without Drift...")
        spec05 = ScenarioSpec(
            scenario_type=DriftScenarioType.ATTACK_ONLY,
            attack_start_step=60,
            attack_end_step=95,
            attack_type="intercept_resend",
            total_steps=self.config.total_steps,
            seed=105,
        )
        df05, _ = DriftScenarioGenerator(spec05).generate_stream()
        engine05, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE)
        res05 = engine05.stream_trajectory(df05[feature_cols], timestamps_s=df05["timestamp"].values)
        m05 = self.evaluator.evaluate(res05, df05, "exp_05", "attack_only", "safe")
        all_metrics.append(m05)
        manifest["experiments"]["exp_05"] = asdict(m05)

        # -------------------------------------------------------------
        # EXPERIMENT 06: Drift Followed by Attack (A -> B -> Attack in B)
        # -------------------------------------------------------------
        logger.info("Executing Experiment 06: Drift Followed by Attack...")
        spec06 = ScenarioSpec(
            scenario_type=DriftScenarioType.DRIFT_THEN_ATTACK,
            drift_start_step=30,
            attack_start_step=90,
            attack_end_step=125,
            attack_type="intercept_resend",
            total_steps=self.config.total_steps,
            seed=106,
        )
        df06, _ = DriftScenarioGenerator(spec06).generate_stream()
        engine06, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE)
        res06 = engine06.stream_trajectory(df06[feature_cols], timestamps_s=df06["timestamp"].values)
        m06 = self.evaluator.evaluate(res06, df06, "exp_06", "drift_then_attack", "safe")
        all_metrics.append(m06)
        manifest["experiments"]["exp_06"] = asdict(m06)

        plot_drift_telemetry_timeline(
            df06, res06, self.plots_dir / "exp06_drift_then_attack_timeline.png",
            title="Experiment 06: Benign Drift Followed by Attack in Condition B"
        )

        # -------------------------------------------------------------
        # EXPERIMENT 07: Attack During Drift (Stress Test)
        # -------------------------------------------------------------
        logger.info("Executing Experiment 07: Attack During Drift...")
        spec07 = ScenarioSpec(
            scenario_type=DriftScenarioType.ATTACK_DURING_DRIFT,
            drift_start_step=35,
            drift_end_step=80,
            total_steps=self.config.total_steps,
            seed=107,
        )
        df07, _ = DriftScenarioGenerator(spec07).generate_stream()
        engine07, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE)
        res07 = engine07.stream_trajectory(df07[feature_cols], timestamps_s=df07["timestamp"].values)
        m07 = self.evaluator.evaluate(res07, df07, "exp_07", "attack_during_drift", "safe")
        all_metrics.append(m07)
        manifest["experiments"]["exp_07"] = asdict(m07)

        # -------------------------------------------------------------
        # EXPERIMENT 08: Fixed vs Naive vs Safe Adaptation (HEADLINE COMPARISON)
        # -------------------------------------------------------------
        logger.info("Executing Experiment 08: Fixed vs Naive vs Safe Adaptation...")
        spec08 = ScenarioSpec(
            scenario_type=DriftScenarioType.CONTAMINATION_STRESS,
            drift_start_step=30,
            drift_end_step=75,
            attack_type="intercept_resend",
            total_steps=self.config.total_steps,
            seed=108,
        )
        df08, _ = DriftScenarioGenerator(spec08).generate_stream()

        # 8A: Fixed policy (No adaptation)
        eng_fixed, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.FIXED)
        res_fixed = eng_fixed.stream_trajectory(df08[feature_cols], timestamps_s=df08["timestamp"].values)
        m08_fixed = self.evaluator.evaluate(res_fixed, df08, "exp_08_fixed", "contamination_stress", "fixed")

        # 8B: Naive policy (Unconstrained immediate adaptation)
        eng_naive, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.NAIVE)
        res_naive = eng_naive.stream_trajectory(df08[feature_cols], timestamps_s=df08["timestamp"].values)
        m08_naive = self.evaluator.evaluate(res_naive, df08, "exp_08_naive", "contamination_stress", "naive")

        # 8C: Safe policy (Gated, shadow-validated adaptation)
        eng_safe, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE)
        res_safe = eng_safe.stream_trajectory(df08[feature_cols], timestamps_s=df08["timestamp"].values)
        m08_safe = self.evaluator.evaluate(res_safe, df08, "exp_08_safe", "contamination_stress", "safe")

        all_metrics.extend([m08_fixed, m08_naive, m08_safe])
        manifest["experiments"]["exp_08_fixed"] = asdict(m08_fixed)
        manifest["experiments"]["exp_08_naive"] = asdict(m08_naive)
        manifest["experiments"]["exp_08_safe"] = asdict(m08_safe)

        # Plot Comparative Analysis
        comp_df = pd.DataFrame([
            {"policy": "Fixed Baseline", "fpr": m08_fixed.fpr, "recall": m08_fixed.recall, "f1": m08_fixed.f1},
            {"policy": "Naive Adaptive", "fpr": m08_naive.fpr, "recall": m08_naive.recall, "f1": m08_naive.f1},
            {"policy": "Safe Adaptive", "fpr": m08_safe.fpr, "recall": m08_safe.recall, "f1": m08_safe.f1},
        ])
        plot_fixed_vs_naive_vs_safe(
            comp_df, self.plots_dir / "exp08_fixed_vs_naive_vs_safe.png",
            title="Experiment 08: Fixed vs Naive vs Safe Adaptation Security Comparison"
        )

        # -------------------------------------------------------------
        # EXPERIMENT 09: Drift Detector Comparison (Wasserstein vs KS vs PSI)
        # -------------------------------------------------------------
        logger.info("Executing Experiment 09: Drift Detector Comparison...")
        for metric_type in (DriftMetric.WASSERSTEIN, DriftMetric.KOLMOGOROV_SMIRNOV, DriftMetric.PSI):
            eng_m, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE, drift_metric=metric_type)
            res_m = eng_m.stream_trajectory(df02[feature_cols], timestamps_s=df02["timestamp"].values)
            m_comp = self.evaluator.evaluate(res_m, df02, f"exp_09_{metric_type.value}", "abrupt_benign", f"safe_{metric_type.value}")
            all_metrics.append(m_comp)
            manifest["experiments"][f"exp_09_{metric_type.value}"] = asdict(m_comp)

        # -------------------------------------------------------------
        # EXPERIMENT 10: Adaptation Window Study (Short=15 vs Med=30 vs Long=50)
        # -------------------------------------------------------------
        logger.info("Executing Experiment 10: Adaptation Window Study...")
        for win_len in (15, 30, 50):
            eng_w, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE, cand_buffer_size=win_len, val_window=win_len // 2)
            res_w = eng_w.stream_trajectory(df03[feature_cols], timestamps_s=df03["timestamp"].values)
            m_win = self.evaluator.evaluate(res_w, df03, f"exp_10_win_{win_len}", "gradual_benign", f"safe_win_{win_len}")
            all_metrics.append(m_win)
            manifest["experiments"][f"exp_10_win_{win_len}"] = asdict(m_win)

        # -------------------------------------------------------------
        # EXPERIMENT 11: Failure Recovery & Rollback Verification
        # -------------------------------------------------------------
        logger.info("Executing Experiment 11: Failure Recovery & Rollback Verification...")
        eng11, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE)
        # Step through 50 steps
        for i in range(50):
            eng11.update(df06[feature_cols].iloc[i].values, timestamp_s=float(df06["timestamp"].iloc[i]))

        # Save checkpoint
        chk_dir = self.output_dir / "checkpoints" / "exp11_checkpoint"
        eng11.save_checkpoint(chk_dir)

        # Reload state in fresh engine
        eng11_resumed, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE)
        eng11_resumed.load_checkpoint(chk_dir)
        # Complete remaining trajectory
        res11_tail: list[Any] = []
        for i in range(50, len(df06)):
            r = eng11_resumed.update(df06[feature_cols].iloc[i].values, timestamp_s=float(df06["timestamp"].iloc[i]))
            res11_tail.append(r)

        m11 = self.evaluator.evaluate(res11_tail, df06.iloc[50:].reset_index(drop=True), "exp_11", "failure_recovery", "safe")
        all_metrics.append(m11)
        manifest["experiments"]["exp_11"] = asdict(m11)

        # -------------------------------------------------------------
        # EXPERIMENT 12: Out-of-Distribution (OOD) Normal Generalization
        # -------------------------------------------------------------
        logger.info("Executing Experiment 12: OOD Generalization...")
        spec12 = ScenarioSpec(
            scenario_type=DriftScenarioType.OOD_GENERALIZATION,
            drift_start_step=35,
            channel_loss_delta_db=4.5,
            temp_delta_c=12.0,
            total_steps=self.config.total_steps,
            seed=112,
        )
        df12, _ = DriftScenarioGenerator(spec12).generate_stream()
        engine12, _ = self._setup_engine(X_train_raw, feature_cols, policy=AdaptationPolicy.SAFE)
        res12 = engine12.stream_trajectory(df12[feature_cols], timestamps_s=df12["timestamp"].values)
        m12 = self.evaluator.evaluate(res12, df12, "exp_12", "ood_generalization", "safe")
        all_metrics.append(m12)
        manifest["experiments"]["exp_12"] = asdict(m12)

        # Extra Plots: Distribution comparison & Drift Heatmap
        ref_samples_clean = X_train_raw[:50]
        drifted_samples_clean = df02[feature_cols].values[70:120]
        plot_distribution_comparison(
            reference_data=ref_samples_clean,
            drifted_data=drifted_samples_clean,
            feature_names=feature_cols[:4],
            output_path=self.plots_dir / "distribution_comparison.png",
        )

        # Generate feature drift matrix for heatmap
        drift_heatmap_matrix = np.zeros((len(feature_cols), min(50, len(res03))))
        for step_idx in range(drift_heatmap_matrix.shape[1]):
            # Pseudo distance per feature for visualization
            sub_ref = ref_samples_clean
            sub_curr = df03[feature_cols].values[max(0, step_idx*3-15):max(1, step_idx*3+5)]
            if len(sub_curr) >= 2:
                for f_idx in range(len(feature_cols)):
                    dist = abs(np.mean(sub_curr[:, f_idx]) - np.mean(sub_ref[:, f_idx])) / (np.std(sub_ref[:, f_idx]) + 1e-4)
                    drift_heatmap_matrix[f_idx, step_idx] = dist

        plot_drift_heatmap(
            feature_drift_matrix=drift_heatmap_matrix,
            feature_names=feature_cols,
            window_indices=list(range(drift_heatmap_matrix.shape[1])),
            output_path=self.plots_dir / "drift_heatmap.png",
        )

        # Serialize summary.csv and manifest.json
        summary_df = pd.DataFrame([asdict(m) for m in all_metrics])
        summary_csv_path = self.output_dir / "summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"V0.8 Experiment suite completed. Output stored in {self.output_dir}")
        return {
            "summary_csv": str(summary_csv_path),
            "manifest_json": str(manifest_path),
            "total_experiments": len(all_metrics),
            "metrics": all_metrics,
        }
