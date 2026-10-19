"""End-to-end temporal experiment runner and benchmarking suite (V0.7).

Orchestrates:
1. Scenario-isolated data loading and feature extraction
2. Strictly causal temporal windowing (terminal, any, majority labeling)
3. Preprocessing fit strictly on normal training partitions
4. Model training (Point baselines, Rolling Z-score, EWMA, GRU/LSTM Sequence Autoencoders)
5. Threshold calibration on normal validation partitions
6. Temporal alarm state filtering (K-of-N persistence, hysteresis, cooldown)
7. Dual sample-level and event-level evaluation (EDR, Detection Delay, FAR/hour)
8. Window length and persistence ablation studies
9. Streaming causal inference latency profiling
10. OOD generalization gap evaluation
11. Publication-grade visualization generation and manifest serialization
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

from qism.ml.data.loader import MLDatasetLoader
from qism.ml.features.extractor import FeatureExtractor
from qism.ml.models.registry import build_model
from qism.ml.preprocessing.scalers import PreprocessorPipeline
from qism.ml.thresholding.calibration import ThresholdCalibrator
from qism.random import seed_all
from qism.temporal.alarm import AlarmConfig, TemporalAlarmStateMachine
from qism.temporal.evaluation import TemporalEvaluator
from qism.temporal.features import TemporalFeatureConfig, TemporalFeatureEngine
from qism.temporal.models import (
    EWMAAnomalyDetector,
    RollingZScoreDetector,
    SequenceAutoencoderDetector,
)
from qism.temporal.plots import (
    plot_feature_decomposition,
    plot_roc_pr_comparison,
    plot_score_distribution_comparison,
    plot_temporal_monitoring_timeline,
    plot_window_length_study,
)
from qism.temporal.streaming import StreamingConfig, StreamingInferenceEngine
from qism.temporal.windowing import LabelingStrategy, TemporalWindowBuilder, TemporalWindowConfig

logger = logging.getLogger(__name__)


@dataclass
class TemporalBenchmarkConfig:
    """Configuration for temporal benchmark runs."""
    dataset_root: str = "data/v0.5_small/v05_small_dataset"
    output_dir: str = "outputs/v07/experiments"
    master_seed: int = 42
    feature_set: str = "full"
    window_length: int = 5

    stride: int = 1
    target_fpr: float = 0.05
    k_persistence: int = 2
    n_window: int = 3
    cooldown_steps: int = 2
    scaler: str = "robust"
    repetition_rate_hz: float = 1_000_000.0
    pulses_per_window: int = 2000
    device: str = "cpu"
    epochs: int = 15
    batch_size: int = 16


class TemporalExperimentRunner:
    """Executes the complete V0.7 temporal experiment suite."""

    def __init__(self, config: TemporalBenchmarkConfig | None = None):
        self.config = config or TemporalBenchmarkConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        seed_all(self.config.master_seed)

    def load_and_prepare_data(self) -> dict[str, Any]:
        """Load datasets and extract observable features."""
        logger.info(f"Loading dataset from {self.config.dataset_root}")
        loader = MLDatasetLoader(
            dataset_root=self.config.dataset_root,
            master_seed=self.config.master_seed,
            train_val_split_ratio=0.70,
        )

        df_train_raw = loader.load_split_dataframe("train")
        df_val_raw = loader.load_split_dataframe("val")
        df_test_raw = loader.load_split_dataframe("test")
        df_gen_raw = loader.load_split_dataframe("generalization")

        extractor = FeatureExtractor(
            feature_set=self.config.feature_set,
            add_temporal_features=False,
        )

        X_train_df, y_train_df = extractor.extract(df_train_raw)
        X_val_df, y_val_df = extractor.extract(df_val_raw)
        X_test_df, y_test_df = extractor.extract(df_test_raw)
        X_gen_df, y_gen_df = extractor.extract(df_gen_raw)

        return {
            "loader": loader,
            "extractor": extractor,
            "train": (X_train_df, y_train_df),
            "val": (X_val_df, y_val_df),
            "test": (X_test_df, y_test_df),
            "gen": (X_gen_df, y_gen_df),
        }

    def run_benchmark(self) -> dict[str, Any]:
        """Run full benchmark matrix across models, window lengths, persistence, and streaming."""
        start_time = datetime.now(timezone.utc)
        data = self.load_and_prepare_data()

        X_train_df, y_train_df = data["train"]
        X_val_df, y_val_df = data["val"]
        X_test_df, y_test_df = data["test"]
        X_gen_df, y_gen_df = data["gen"]

        # 1. Preprocessing fit strictly on normal training set
        preprocessor = PreprocessorPipeline(scaler=self.config.scaler)
        X_train_scaled = preprocessor.fit_transform(X_train_df)
        X_val_scaled = preprocessor.transform(X_val_df)
        X_test_scaled = preprocessor.transform(X_test_df)
        X_gen_scaled = preprocessor.transform(X_gen_df)

        feature_names = list(X_train_df.columns)

        # 2. Build temporal windows for default window length
        win_cfg = TemporalWindowConfig(
            window_length=self.config.window_length,
            stride=self.config.stride,
            causal_padding=True,
            labeling_strategy=LabelingStrategy.TERMINAL,
        )
        builder = TemporalWindowBuilder(win_cfg)

        X_train_win, meta_train, y_train_win = builder.build_windows(X_train_scaled, y_train_df)
        X_val_win, meta_val, y_val_win = builder.build_windows(X_val_scaled, y_val_df)
        X_test_win, meta_test, y_test_win = builder.build_windows(X_test_scaled, y_test_df)
        X_gen_win, meta_gen, y_gen_win = builder.build_windows(X_gen_scaled, y_gen_df)

        evaluator = TemporalEvaluator(
            repetition_rate_hz=self.config.repetition_rate_hz,
            default_pulses_per_window=self.config.pulses_per_window,
        )

        test_meta_df = builder.to_metadata_dataframe(meta_test)
        gen_meta_df = builder.to_metadata_dataframe(meta_gen)

        results: dict[str, Any] = {}
        summary_rows: list[dict[str, Any]] = []

        # =========================================================================
        # Experiment 1: V0.6 Point Baseline (Isolation Forest on terminal state)
        # =========================================================================
        logger.info("Evaluating Experiment 1: Point Baseline (Isolation Forest)...")
        pt_model = build_model("isolation_forest", random_state=self.config.master_seed)
        X_tr_arr = X_train_scaled if isinstance(X_train_scaled, np.ndarray) else X_train_scaled.values
        X_va_arr = X_val_scaled if isinstance(X_val_scaled, np.ndarray) else X_val_scaled.values
        pt_model.fit(X_tr_arr)
        pt_val_scores = pt_model.score(X_va_arr)
        pt_thresh = ThresholdCalibrator(target_fpr=self.config.target_fpr).calibrate(pt_val_scores)["threshold"]
        pt_model.set_threshold(pt_thresh)

        # Score on test windows using terminal state
        X_test_terminal = X_test_win[:, -1, :]
        pt_test_scores = pt_model.score(X_test_terminal)
        pt_raw_alarms = (pt_test_scores >= pt_thresh).astype(int)

        # Alarm state machine
        alarm_sm = TemporalAlarmStateMachine(
            AlarmConfig(
                threshold=pt_thresh,
                k_consecutive=self.config.k_persistence,
                n_window=self.config.n_window,
                cooldown_steps=self.config.cooldown_steps,
            )
        )
        _, pt_filt_alarms, _ = alarm_sm.process_sequence(pt_test_scores)

        pt_summary = evaluator.evaluate(
            scores=pt_test_scores,
            raw_alarms=pt_raw_alarms,
            filtered_alarms=pt_filt_alarms,
            y_true_active=y_test_win,
            y_meta_df=test_meta_df,
        )
        results["exp1_point_baseline"] = pt_summary
        summary_rows.append(
            self._make_summary_row(
                "Exp 1: Point Baseline (IF)",
                "Point (d)",
                False,
                pt_summary,
            )
        )

        # =========================================================================
        # Experiment 2: Temporal Statistical Baseline (Rolling Z-Score)
        # =========================================================================
        logger.info("Evaluating Experiment 2: Rolling Z-Score Baseline...")
        z_model = RollingZScoreDetector(window_length=self.config.window_length)
        z_model.fit(X_train_win)
        z_val_scores = z_model.score(X_val_win)
        z_thresh = ThresholdCalibrator(target_fpr=self.config.target_fpr).calibrate(z_val_scores)["threshold"]
        z_model.set_threshold(z_thresh)

        z_test_scores = z_model.score(X_test_win)
        z_raw_alarms = (z_test_scores >= z_thresh).astype(int)
        z_sm = TemporalAlarmStateMachine(
            AlarmConfig(
                threshold=z_thresh,
                k_consecutive=self.config.k_persistence,
                n_window=self.config.n_window,
                cooldown_steps=self.config.cooldown_steps,
            )
        )
        _, z_filt_alarms, _ = z_sm.process_sequence(z_test_scores)

        z_summary = evaluator.evaluate(
            scores=z_test_scores,
            raw_alarms=z_raw_alarms,
            filtered_alarms=z_filt_alarms,
            y_true_active=y_test_win,
            y_meta_df=test_meta_df,
        )
        results["exp2_rolling_zscore"] = z_summary
        summary_rows.append(
            self._make_summary_row(
                "Exp 2: Rolling Z-Score",
                f"Window (L={self.config.window_length}, d)",
                True,
                z_summary,
            )
        )

        # =========================================================================
        # Experiment 3: Temporal Statistical Baseline (EWMA Anomaly Detector)
        # =========================================================================
        logger.info("Evaluating Experiment 3: EWMA Anomaly Baseline...")
        ewma_model = EWMAAnomalyDetector(alpha=0.3)
        ewma_model.fit(X_train_win)
        ewma_val_scores = ewma_model.score(X_val_win)
        ewma_thresh = ThresholdCalibrator(target_fpr=self.config.target_fpr).calibrate(ewma_val_scores)["threshold"]
        ewma_model.set_threshold(ewma_thresh)

        ewma_test_scores = ewma_model.score(X_test_win)
        ewma_raw_alarms = (ewma_test_scores >= ewma_thresh).astype(int)
        ewma_sm = TemporalAlarmStateMachine(
            AlarmConfig(
                threshold=ewma_thresh,
                k_consecutive=self.config.k_persistence,
                n_window=self.config.n_window,
                cooldown_steps=self.config.cooldown_steps,
            )
        )
        _, ewma_filt_alarms, _ = ewma_sm.process_sequence(ewma_test_scores)
        ewma_summary = evaluator.evaluate(
            scores=ewma_test_scores,
            raw_alarms=ewma_raw_alarms,
            filtered_alarms=ewma_filt_alarms,
            y_true_active=y_test_win,
            y_meta_df=test_meta_df,
        )
        results["exp3_ewma"] = ewma_summary
        summary_rows.append(
            self._make_summary_row(
                "Exp 3: EWMA Baseline",
                f"Window (L={self.config.window_length}, d)",
                True,
                ewma_summary,
            )
        )

        # =========================================================================
        # Experiment 4: Sequence Autoencoder (GRU)
        # =========================================================================
        logger.info("Evaluating Experiment 4: Sequence Autoencoder (GRU)...")
        gru_ae = SequenceAutoencoderDetector(
            sequence_length=self.config.window_length,
            feature_dim=X_train_win.shape[2],
            rnn_type="GRU",
            hidden_dim=32,
            latent_dim=16,
            num_layers=1,
            learning_rate=1e-3,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            device=self.config.device,
            seed=self.config.master_seed,
        )
        gru_ae.fit(X_train_win)
        gru_val_scores = gru_ae.score(X_val_win)
        gru_thresh = ThresholdCalibrator(target_fpr=self.config.target_fpr).calibrate(gru_val_scores)["threshold"]
        gru_ae.set_threshold(gru_thresh)

        gru_test_scores = gru_ae.score(X_test_win)
        gru_raw_alarms = (gru_test_scores >= gru_thresh).astype(int)
        gru_sm = TemporalAlarmStateMachine(
            AlarmConfig(
                threshold=gru_thresh,
                k_consecutive=self.config.k_persistence,
                n_window=self.config.n_window,
                cooldown_steps=self.config.cooldown_steps,
            )
        )
        _, gru_filt_alarms, _ = gru_sm.process_sequence(gru_test_scores)
        gru_summary = evaluator.evaluate(
            scores=gru_test_scores,
            raw_alarms=gru_raw_alarms,
            filtered_alarms=gru_filt_alarms,
            y_true_active=y_test_win,
            y_meta_df=test_meta_df,
        )
        results["exp4_gru_ae"] = gru_summary
        summary_rows.append(
            self._make_summary_row(
                "Exp 4: Sequence AE (GRU)",
                f"Sequence (L={self.config.window_length}, d)",
                True,
                gru_summary,
            )
        )

        # =========================================================================
        # Experiment 5: Sequence Autoencoder (LSTM)
        # =========================================================================
        logger.info("Evaluating Experiment 5: Sequence Autoencoder (LSTM)...")
        lstm_ae = SequenceAutoencoderDetector(
            sequence_length=self.config.window_length,
            feature_dim=X_train_win.shape[2],
            rnn_type="LSTM",
            hidden_dim=32,
            latent_dim=16,
            num_layers=1,
            learning_rate=1e-3,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            device=self.config.device,
            seed=self.config.master_seed,
        )
        lstm_ae.fit(X_train_win)
        lstm_val_scores = lstm_ae.score(X_val_win)
        lstm_thresh = ThresholdCalibrator(target_fpr=self.config.target_fpr).calibrate(lstm_val_scores)["threshold"]
        lstm_ae.set_threshold(lstm_thresh)

        lstm_test_scores = lstm_ae.score(X_test_win)
        lstm_raw_alarms = (lstm_test_scores >= lstm_thresh).astype(int)
        lstm_sm = TemporalAlarmStateMachine(
            AlarmConfig(
                threshold=lstm_thresh,
                k_consecutive=self.config.k_persistence,
                n_window=self.config.n_window,
                cooldown_steps=self.config.cooldown_steps,
            )
        )
        _, lstm_filt_alarms, _ = lstm_sm.process_sequence(lstm_test_scores)

        lstm_summary = evaluator.evaluate(
            scores=lstm_test_scores,
            raw_alarms=lstm_raw_alarms,
            filtered_alarms=lstm_filt_alarms,
            y_true_active=y_test_win,
            y_meta_df=test_meta_df,
        )
        results["exp5_lstm_ae"] = lstm_summary
        summary_rows.append(
            self._make_summary_row(
                "Exp 5: Sequence AE (LSTM)",
                f"Sequence (L={self.config.window_length}, d)",
                True,
                lstm_summary,
            )
        )

        # =========================================================================
        # Experiment 6: Temporal Alarm Persistence Ablation (Raw vs K=1 vs K=2 vs K=3)
        # =========================================================================
        logger.info("Evaluating Experiment 6: Persistence & Hysteresis Ablation...")
        persist_configs = [
            ("Raw Thresholding (K=1/1)", 1, 1, 0.0),
            ("Persistence K=2 of 3", 2, 3, 0.0),
            ("Persistence K=3 of 5", 3, 5, 0.0),
            ("Hysteresis + Persistence", 2, 3, 0.75 * gru_thresh),
        ]
        persist_results = {}
        for name, k_c, n_w, rec_t in persist_configs:
            sm = TemporalAlarmStateMachine(
                AlarmConfig(
                    threshold=gru_thresh,
                    recovery_threshold=rec_t if rec_t > 0.0 else gru_thresh,
                    k_consecutive=k_c,
                    n_window=n_w,
                    cooldown_steps=self.config.cooldown_steps,
                )
            )
            _, filtered, _ = sm.process_sequence(gru_test_scores)
            res = evaluator.evaluate(
                scores=gru_test_scores,
                raw_alarms=gru_raw_alarms,
                filtered_alarms=filtered,
                y_true_active=y_test_win,
                y_meta_df=test_meta_df,
            )
            persist_results[name] = res
            summary_rows.append(
                self._make_summary_row(
                    f"Exp 6: Alarm - {name}",
                    f"GRU AE ({name})",
                    True,
                    res,
                )
            )
        results["exp6_persistence_ablation"] = persist_results

        # =========================================================================
        # Experiment 7: Window Length Study (L = 2, 3, 5)
        # =========================================================================
        logger.info("Evaluating Experiment 7: Window Length Ablation...")
        win_lengths = [2, 3, 5]
        win_recalls = []
        win_fprs = []
        win_delays = []
        window_results = {}

        for L in win_lengths:
            w_b = TemporalWindowBuilder(
                TemporalWindowConfig(
                    window_length=L,
                    stride=1,
                    causal_padding=True,
                    labeling_strategy=LabelingStrategy.TERMINAL,
                )
            )
            X_tr_w, _, _ = w_b.build_windows(X_train_scaled, y_train_df)
            X_va_w, _, _ = w_b.build_windows(X_val_scaled, y_val_df)
            X_te_w, m_te, y_te_w = w_b.build_windows(X_test_scaled, y_test_df)
            m_te_df = w_b.to_metadata_dataframe(m_te)

            m_l = SequenceAutoencoderDetector(
                sequence_length=L,
                feature_dim=X_tr_w.shape[2],
                rnn_type="GRU",
                hidden_dim=32,
                latent_dim=16,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                device=self.config.device,
                seed=self.config.master_seed,
            )
            m_l.fit(X_tr_w)
            v_sc = m_l.score(X_va_w)
            th = ThresholdCalibrator(target_fpr=self.config.target_fpr).calibrate(v_sc)["threshold"]
            m_l.set_threshold(th)

            te_sc = m_l.score(X_te_w)
            raw_al = (te_sc >= th).astype(int)
            sm = TemporalAlarmStateMachine(AlarmConfig(threshold=th, k_consecutive=2, n_window=3))
            _, filt_al, _ = sm.process_sequence(te_sc)
            w_eval = evaluator.evaluate(
                scores=te_sc,
                raw_alarms=raw_al,
                filtered_alarms=filt_al,
                y_true_active=y_te_w,
                y_meta_df=m_te_df,
            )
            window_results[f"L={L}"] = w_eval
            win_recalls.append(w_eval.filtered_window_metrics.get("recall", 0.0))
            win_fprs.append(w_eval.filtered_window_metrics.get("fpr", 0.0))
            med_delay = w_eval.detection_delay_summary.get("median_delay_s", 0.0)
            win_delays.append(med_delay if med_delay is not None else 0.0)
            summary_rows.append(
                self._make_summary_row(
                    f"Exp 7: Window Study (L={L})",
                    f"GRU AE (L={L})",
                    True,
                    w_eval,
                )
            )

        results["exp7_window_study"] = window_results

        # Plot window study
        plot_window_length_study(
            lengths=win_lengths,
            recalls=win_recalls,
            fprs=win_fprs,
            median_delays_s=win_delays,
            output_path=self.plots_dir / "window_length_study.png",
        )

        # =========================================================================
        # Experiment 8: Streaming Causal Inference & Latency Benchmark
        # =========================================================================
        logger.info("Evaluating Experiment 8: Streaming Inference & Latency...")
        stream_engine = StreamingInferenceEngine(
            model=gru_ae,
            preprocessor=preprocessor,
            alarm_machine=gru_sm,
            history_window_length=self.config.window_length,
        )

        stream_latencies_us: list[float] = []
        streaming_alarms: list[int] = []
        stream_scores: list[float] = []

        # Run causal sample-by-sample ingestion
        for i in range(len(X_test_scaled)):
            sample_t = X_test_scaled[i]
            res = stream_engine.update(sample_t, timestamp_s=float(i * 0.002))
            stream_latencies_us.append(res.computational_latency_ms * 1000.0)
            streaming_alarms.append(int(res.filtered_alarm))
            stream_scores.append(res.score)

        p95_lat = float(np.percentile(stream_latencies_us, 95))
        mean_lat = float(np.mean(stream_latencies_us))
        logger.info(f"Streaming latency: Mean={mean_lat:.1f}us, P95={p95_lat:.1f}us")

        results["exp8_streaming"] = {
            "mean_latency_us": mean_lat,
            "p50_latency_us": float(np.percentile(stream_latencies_us, 50)),
            "p95_latency_us": p95_lat,
            "max_latency_us": float(np.max(stream_latencies_us)),
            "total_streamed_steps": len(stream_latencies_us),
        }

        # =========================================================================
        # Experiment 9: OOD Generalization Evaluation
        # =========================================================================
        logger.info("Evaluating Experiment 9: OOD Generalization...")
        if len(X_gen_win) > 0:
            gen_scores = gru_ae.score(X_gen_win)
            gen_raw_alarms = (gen_scores >= gru_thresh).astype(int)
            gen_sm = TemporalAlarmStateMachine(
                AlarmConfig(
                    threshold=gru_thresh,
                    k_consecutive=self.config.k_persistence,
                    n_window=self.config.n_window,
                )
            )
            _, gen_filt_alarms, _ = gen_sm.process_sequence(gen_scores)

            gen_summary = evaluator.evaluate(
                scores=gen_scores,
                raw_alarms=gen_raw_alarms,
                filtered_alarms=gen_filt_alarms,
                y_true_active=y_gen_win,
                y_meta_df=gen_meta_df,
            )
            results["exp9_generalization"] = gen_summary
            summary_rows.append(
                self._make_summary_row(
                    "Exp 9: OOD Generalization",
                    f"GRU AE (OOD Test)",
                    True,
                    gen_summary,
                )
            )

        # =========================================================================
        # Experiment 10: Feature Reconstruction Error Decomposition (Interpretability)
        # =========================================================================
        logger.info("Evaluating Experiment 10: Feature Decomposition E_j...")
        # Get feature error contributions during attack episodes
        attack_mask = (y_test_win == 1)
        if np.any(attack_mask):
            feat_contribs = gru_ae.feature_error_contributions(X_test_win[attack_mask])
            mean_e_j = np.mean(feat_contribs, axis=0)
            ranked_indices = np.argsort(mean_e_j)[::-1]
            top_features = [
                {"feature": feature_names[i], "mean_error": float(mean_e_j[i])}
                for i in ranked_indices[:10]
            ]
            results["exp10_interpretability"] = top_features

            plot_feature_decomposition(
                feature_names=feature_names,
                contributions=feat_contribs,
                output_path=self.plots_dir / "feature_decomposition.png",
            )

        # =========================================================================
        # Generate Core Visualizations
        # =========================================================================
        logger.info("Generating publication-grade plots...")
        # 1. Timeline plot on representative test sequence
        plot_temporal_monitoring_timeline(
            timestamps_s=np.arange(len(gru_test_scores)) * 0.002,
            scores=gru_test_scores,
            threshold=gru_thresh,
            recovery_threshold=0.75 * gru_thresh,
            raw_alarms=gru_raw_alarms,
            filtered_alarms=gru_filt_alarms,
            attack_active=y_test_win,
            telemetry_samples=X_test_df.iloc[:len(gru_test_scores)],
            output_path=self.plots_dir / "temporal_timeline.png",
        )

        # 2. Score distributions
        norm_scores = gru_test_scores[y_test_win == 0]
        atk_scores = gru_test_scores[y_test_win == 1]
        plot_score_distribution_comparison(
            normal_scores=norm_scores,
            attack_scores=atk_scores,
            threshold=gru_thresh,
            output_path=self.plots_dir / "score_distributions.png",
        )

        # 3. ROC / PR comparison: Point Baseline vs GRU Sequence AE
        plot_roc_pr_comparison(
            y_true=y_test_win,
            point_scores=pt_test_scores,
            temporal_scores=gru_test_scores,
            point_name="V0.6 Point Baseline (IF)",
            temporal_name="V0.7 GRU Sequence AE",
            output_path=self.plots_dir / "roc_pr_comparison.png",
        )

        # Save summary CSV
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = self.output_dir / "summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Summary saved to {summary_csv_path}")

        # Save experiment manifest
        manifest = {
            "version": "0.7.0",
            "timestamp": start_time.isoformat(),
            "config": asdict(self.config),
            "total_runtime_s": (datetime.now(timezone.utc) - start_time).total_seconds(),
            "streaming_performance": results.get("exp8_streaming", {}),
            "top_contributing_features": results.get("exp10_interpretability", []),
            "summary_table": summary_rows,
        }
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved to {manifest_path}")

        return {
            "manifest": manifest,
            "summary_df": summary_df,
            "results": results,
        }

    def _make_summary_row(
        self,
        method: str,
        representation: str,
        causal: bool,
        eval_summary: Any,
    ) -> dict[str, Any]:
        """Format an evaluation summary into a clean row for the summary table."""
        raw_m = eval_summary.raw_window_metrics
        filt_m = eval_summary.filtered_window_metrics
        curves = eval_summary.curve_metrics
        delay_sum = eval_summary.detection_delay_summary

        return {
            "Method": method,
            "Representation": representation,
            "Causal": causal,
            "AUROC": round(float(curves.get("auroc", 0.0)), 4),
            "AUPRC": round(float(curves.get("auprc", 0.0)), 4),
            "F1": round(float(filt_m.get("f1", 0.0)), 4),
            "Recall": round(float(filt_m.get("recall", 0.0)), 4),
            "Precision": round(float(filt_m.get("precision", 0.0)), 4),
            "FAR (FPR)": round(float(filt_m.get("fpr", 0.0)), 4),
            "FA / Hour": round(float(eval_summary.false_alarms_per_hour), 2),
            "Median Delay (s)": round(float(delay_sum.get("median_delay_s", 0.0) or 0.0), 4),
            "Episode Recall (EDR)": round(float(eval_summary.episode_detection_rate), 4),
            "Score Separation": round(float(eval_summary.score_separation_ratio), 2),
        }
