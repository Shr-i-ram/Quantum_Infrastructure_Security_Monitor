"""Unified command-line interface for QISM Simulator (Requirements 117–122).

Commands:
    python -m qism generate-scenarios <config>
    python -m qism generate-dataset <config> [--workers N] [--resume] [--fresh]
    python -m qism resume <config>
    python -m qism validate-dataset <dataset_path>... [--check-split]
    python -m qism benchmark <config>
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

from qism import __version__, verify_python_version
from qism.config import load_v05_dataset_config
from qism.datasets.pipeline import DatasetPipeline

log = logging.getLogger("qism.cli")


def cmd_generate_scenarios(args: argparse.Namespace) -> int:
    """Generate and validate population scenarios without running simulations (Requirement 118)."""
    cfg = load_v05_dataset_config(args.config)
    pipeline = DatasetPipeline(cfg)
    result = pipeline.dry_run()
    print("\n" + "=" * 70)
    print(f"QISM V0.5 Scenario Preview — Dataset '{cfg.dataset_id}'")
    print("=" * 70)
    print(f"Status              : {result['status']}")
    print(f"Scenarios Generated : {result['scenarios_generated']}")
    print(f"Total Runs Planned  : {result['workload']['total_runs']}")
    print(f"Total Pulses        : {result['workload']['total_pulses']:,}")
    print(f"Estimated Windows   : {result['workload']['total_estimated_windows']:,}")
    print(f"Estimated Storage   : ~{result['workload']['estimated_storage_mb']} MB")
    print(f"Sampling Strategy   : {result['workload']['sampling_strategy']}")
    print("=" * 70)
    return 0


def cmd_generate_dataset(args: argparse.Namespace) -> int:
    """Execute end-to-end dataset generation (Requirement 119)."""
    cfg = load_v05_dataset_config(args.config)
    if args.workers is not None:
        cfg.worker_count = int(args.workers)
    pipeline = DatasetPipeline(cfg)
    summary = pipeline.run(resume=args.resume, fresh=args.fresh)
    print("\n" + "=" * 70)
    print(f"QISM V0.5 Dataset Generation Summary — '{summary['dataset_id']}'")
    print("=" * 70)
    print(f"Output Directory : {summary['output_dir']}")
    print(f"Total Scenarios  : {summary['total_scenarios']}")
    print(f"Completed Runs   : {summary['completed_runs']}")
    print(f"Failed Runs      : {summary['failed_runs']}")
    print(f"Execution Time   : {summary['wall_time_s']:.2f} s")
    print("=" * 70)
    return 0 if summary["failed_runs"] == 0 else 1


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume incomplete dataset generation (Requirement 120)."""
    args.resume = True
    args.fresh = False
    return cmd_generate_dataset(args)


def cmd_validate_dataset(args: argparse.Namespace) -> int:
    """Validate dataset directories for schema compliance and zero leakage (Requirement 121)."""
    # Delegate to scripts/validate_dataset logic
    import subprocess
    cmd = [sys.executable, "scripts/validate_dataset.py"] + args.paths
    if args.check_split:
        cmd.append("--check-split")
    res = subprocess.run(cmd)
    return res.returncode


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run serial vs parallel throughput benchmark (Requirement 122)."""
    cfg = load_v05_dataset_config(args.config)
    print("\n" + "=" * 70)
    print(f"QISM V0.5 Scalability Benchmark — Dataset '{cfg.dataset_id}'")
    print("=" * 70)

    # 1. Serial run (1 worker)
    cfg_serial = load_v05_dataset_config(args.config)
    cfg_serial.dataset_id = f"{cfg.dataset_id}_bench_serial"
    cfg_serial.worker_count = 1
    p_serial = DatasetPipeline(cfg_serial)
    t0 = time.perf_counter()
    res_serial = p_serial.run(fresh=True)
    time_serial = time.perf_counter() - t0

    # 2. Parallel run (configured workers)
    worker_count = args.workers or max(2, cfg.worker_count)
    cfg_parallel = load_v05_dataset_config(args.config)
    cfg_parallel.dataset_id = f"{cfg.dataset_id}_bench_parallel"
    cfg_parallel.worker_count = worker_count
    p_parallel = DatasetPipeline(cfg_parallel)
    t0 = time.perf_counter()
    res_parallel = p_parallel.run(fresh=True)
    time_parallel = time.perf_counter() - t0

    speedup = time_serial / time_parallel if time_parallel > 0 else 1.0

    print(f"Serial Execution (1 worker)     : {time_serial:.2f} s ({res_serial['completed_runs']} runs)")
    print(f"Parallel Execution ({worker_count} workers) : {time_parallel:.2f} s ({res_parallel['completed_runs']} runs)")
    print(f"Measured Speedup                : {speedup:.2f}x")
    print("=" * 70)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train anomaly detection model and calibrate threshold (Requirement 146)."""
    from qism.ml.experiments.config import ExperimentConfig
    from qism.ml.experiments.runner import ExperimentRunner

    cfg = ExperimentConfig.from_yaml(args.config)
    runner = ExperimentRunner(cfg)
    res = runner.run()
    print("\n" + "=" * 70)
    print(f"QISM V0.6 Model Training & Calibration — '{cfg.experiment_id}'")
    print("=" * 70)
    print(f"Model Type          : {cfg.model_type}")
    print(f"Feature Set         : {cfg.feature_set}")
    print(f"Calibrated Threshold: {res['manifest']['calibrated_threshold']:.4f}")
    print(f"Artifacts Saved To  : {runner.exp_dir}")
    print("=" * 70)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate saved model against specified dataset split (Requirement 147)."""
    import json
    from qism.ml.data.loader import MLDatasetLoader
    from qism.ml.evaluation.runner import EvaluationRunner
    from qism.ml.experiments.config import ExperimentConfig
    from qism.ml.features.extractor import FeatureExtractor
    from qism.ml.models.registry import build_model
    from qism.ml.preprocessing.scalers import PreprocessorPipeline

    exp_dir = pathlib.Path(args.experiment_dir)
    cfg = ExperimentConfig.from_yaml(exp_dir / "config.yaml")
    prep = PreprocessorPipeline.load(exp_dir / "preprocessor.json")
    with open(exp_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    threshold = manifest["calibrated_threshold"]

    model_cls = build_model(cfg.model_type)
    model = model_cls.load(exp_dir / "model.bin")

    extractor = FeatureExtractor(
        feature_set=cfg.feature_set,
        add_temporal_features=cfg.add_temporal_features,
        rolling_window_length=cfg.rolling_window_length,
    )

    loader = MLDatasetLoader(
        dataset_root=args.dataset or cfg.dataset_root,
        master_seed=cfg.master_seed,
    )
    df_raw = loader.load_split_dataframe(args.split)

    eval_runner = EvaluationRunner(
        model=model,
        preprocessor=prep,
        extractor=extractor,
        threshold=threshold,
        consecutive_k=cfg.consecutive_k,
    )
    res = eval_runner.evaluate(df_raw, split_name=args.split)

    wm = res["window_metrics"]
    cm = res["curve_metrics"]
    im = res["instance_metrics"]

    print("\n" + "=" * 70)
    print(f"QISM V0.6 Evaluation Results — Split '{args.split}'")
    print("=" * 70)
    print(f"Total Windows        : {res['n_samples']}")
    print(f"Precision            : {wm['precision']:.4f}")
    print(f"Recall               : {wm['recall']:.4f}")
    print(f"F1 Score             : {wm['f1']:.4f}")
    print(f"False Positive Rate  : {wm['fpr']:.4f}")
    print(f"AUROC                : {cm['auroc']:.4f}")
    print(f"AUPRC                : {cm['auprc']:.4f}")
    print(f"Episode Detection    : {im.get('detected_attack_episodes', 0)} / {im.get('total_attack_episodes', 0)}")
    print(f"False Alarms / Hour  : {im.get('false_alarms_per_hour', 0.0):.2f}")
    print("=" * 70)
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """Run full benchmark experiment from YAML config (Requirement 148)."""
    from qism.ml.experiments.config import ExperimentConfig
    from qism.ml.experiments.runner import ExperimentRunner

    cfg = ExperimentConfig.from_yaml(args.config)
    runner = ExperimentRunner(cfg)
    res = runner.run()

    iid_m = res["iid_test"]["window_metrics"]
    ood_m = res["ood_generalization"]["window_metrics"]
    gap = res["generalization_gap"]

    print("\n" + "=" * 70)
    print(f"QISM V0.6 Experiment Summary — '{cfg.experiment_id}'")
    print("=" * 70)
    print(f"Model: {cfg.model_type} | Feature Set: {cfg.feature_set}")
    print(f"IID Test       -> Recall: {iid_m['recall']:.3f} | FPR: {iid_m['fpr']:.3f} | F1: {iid_m['f1']:.3f} | AUROC: {res['iid_test']['curve_metrics']['auroc']:.3f}")
    print(f"OOD Generaliz. -> Recall: {ood_m['recall']:.3f} | FPR: {ood_m['fpr']:.3f} | F1: {ood_m['f1']:.3f} | AUROC: {res['ood_generalization']['curve_metrics']['auroc']:.3f}")
    print(f"OOD Gap        -> Recall Gap: {gap['recall_gap']:+.3f} | FPR Degradation: {gap['fpr_gap']:+.3f}")
    print(f"Sanity Check   -> Shuffled AUROC: {res['sanity_experiments'].get('shuffled_labels', {}).get('auroc', 0.5):.3f}")
    print("=" * 70)
    return 0


def cmd_benchmark_ml(args: argparse.Namespace) -> int:
    """Run multi-model anomaly detection benchmark on identical data (Requirement 155)."""
    import pandas as pd
    from qism.ml.experiments.config import ExperimentConfig
    from qism.ml.experiments.runner import ExperimentRunner

    models = ["mahalanobis", "isolation_forest", "one_class_svm", "autoencoder", "temporal_autoencoder"]
    base_cfg = ExperimentConfig.from_yaml(args.config)
    rows = []

    print("\n" + "=" * 80)
    print(f"QISM V0.6 Baseline Anomaly Detection Benchmark — Dataset '{base_cfg.dataset_root}'")
    print("=" * 80)

    for m_type in models:
        cfg = base_cfg.model_copy()
        cfg.experiment_id = f"{base_cfg.experiment_id}_{m_type}"
        cfg.model_type = m_type
        if m_type == "autoencoder":
            cfg.hyperparameters = {"epochs": 15, "lr": 1e-3}
        elif m_type == "temporal_autoencoder":
            cfg.hyperparameters = {"epochs": 15, "seq_len": 3}
        elif m_type == "isolation_forest":
            cfg.hyperparameters = {"n_estimators": 50}

        try:
            runner = ExperimentRunner(cfg)
            res = runner.run()
            iid = res["iid_test"]["window_metrics"]
            ood = res["ood_generalization"]["window_metrics"]
            curves = res["iid_test"]["curve_metrics"]
            im = res["iid_test"]["instance_metrics"]
            delay = im.get("detection_delay_summary", {}).get("median_delay_s")
            rows.append({
                "Model": m_type,
                "IID FPR": f"{iid['fpr']:.3f}",
                "IID Rec": f"{iid['recall']:.3f}",
                "IID F1": f"{iid['f1']:.3f}",
                "IID AUROC": f"{curves['auroc']:.3f}",
                "OOD Rec": f"{ood['recall']:.3f}",
                "OOD FPR": f"{ood['fpr']:.3f}",
                "Delay (s)": f"{delay:.3f}" if delay is not None else "N/A",
            })
        except Exception as e:
            rows.append({
                "Model": m_type,
                "IID FPR": "ERROR",
                "IID Rec": str(e)[:15],
                "IID F1": "ERROR",
                "IID AUROC": "ERROR",
                "OOD Rec": "ERROR",
                "OOD FPR": "ERROR",
                "Delay (s)": "ERROR",
            })

    df_res = pd.DataFrame(rows)
    print(df_res.to_string(index=False))
    print("=" * 80)
    return 0


def cmd_analyze_results(args: argparse.Namespace) -> int:
    """Analyze and display comparison tables from saved experiments (Requirement 149)."""
    import pandas as pd
    root = pathlib.Path(args.experiments_dir)
    summaries = list(root.rglob("summary.csv"))
    if not summaries:
        print(f"No summary.csv files found in {root}")
        return 1

    frames = [pd.read_csv(s) for s in summaries]
    combined = pd.concat(frames, ignore_index=True)
    print("\n" + "=" * 90)
    print("QISM V0.6 Consolidated Anomaly Detection Benchmark Results")
    print("=" * 90)
    cols = [c for c in ["experiment_id", "model_type", "feature_set", "iid_f1", "iid_recall", "iid_fpr", "iid_auroc", "ood_recall", "ood_fpr", "recall_gap"] if c in combined.columns]
    print(combined[cols].to_string(index=False))
    print("=" * 90)
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    """Generate presentation-quality visualizations (V0.1–V0.5 baseline)."""
    from qism.visualization import VisualizationConfig, generate_all_visualizations
    fmts = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    cfg = VisualizationConfig(
        dataset_path=pathlib.Path(args.dataset),
        output_root=pathlib.Path(args.outdir),
        formats=fmts,
        dpi=args.dpi,
    )
    res = generate_all_visualizations(cfg)
    print(f"\nGenerated {res['total_presentation_figures']} presentation figures in {res['presentation_dir']}")
    return 0


def cmd_v07_benchmark(args: argparse.Namespace) -> int:
    """Run V0.7 temporal anomaly detection benchmark suite."""
    from qism.temporal.experiments import TemporalBenchmarkConfig, TemporalExperimentRunner

    cfg = TemporalBenchmarkConfig(
        dataset_root=args.dataset,
        output_dir=args.outdir,
        master_seed=args.seed,
        window_length=args.window_length,
        epochs=args.epochs,
    )
    runner = TemporalExperimentRunner(cfg)
    res = runner.run_benchmark()

    print("\n" + "=" * 95)
    print("QISM V0.7 Temporal & Streaming Anomaly Detection Benchmark Summary")
    print("=" * 95)
    print(res["summary_df"].to_string(index=False))
    print("=" * 95)
    print(f"Results and plots saved to: {runner.output_dir}")
    print("=" * 95)
    return 0


def cmd_v07_stream(args: argparse.Namespace) -> int:
    """Simulate causal streaming anomaly monitoring on dataset telemetry."""
    from qism.ml.data.loader import MLDatasetLoader
    from qism.ml.features.extractor import FeatureExtractor
    from qism.ml.preprocessing.scalers import PreprocessorPipeline
    from qism.temporal.alarm import AlarmConfig, TemporalAlarmStateMachine
    from qism.temporal.models import RollingZScoreDetector
    from qism.temporal.streaming import StreamingConfig, StreamingInferenceEngine

    loader = MLDatasetLoader(args.dataset)
    df_train = loader.load_split_dataframe("train")
    df_test = loader.load_split_dataframe("test")

    extractor = FeatureExtractor(feature_set="full")
    X_train_df, _ = extractor.extract(df_train)
    X_test_df, y_test_df = extractor.extract(df_test)

    prep = PreprocessorPipeline(scaler="robust")
    X_train_scaled = prep.fit_transform(X_train_df)
    X_test_scaled = prep.transform(X_test_df)

    detector = RollingZScoreDetector(window_length=args.window_length)
    detector.fit(X_train_scaled)

    val_features, _ = extractor.extract(loader.load_split_dataframe("val"))
    val_scaled = prep.transform(val_features)
    val_scores = detector.score(val_scaled)
    from qism.ml.thresholding.calibration import ThresholdCalibrator
    thresh = ThresholdCalibrator(target_fpr=0.05).calibrate(val_scores)["threshold"]
    detector.set_threshold(thresh)

    alarm_sm = TemporalAlarmStateMachine(
        AlarmConfig(threshold=thresh, k_consecutive=2, n_window=3)
    )
    engine = StreamingInferenceEngine(
        model=detector,
        preprocessor=prep,
        alarm_machine=alarm_sm,
        history_window_length=args.window_length,
    )


    print("\n" + "=" * 80)
    print("QISM V0.7 Causal Streaming Monitoring Active (Press Ctrl+C to halt)")
    print("=" * 80)
    n_steps = min(len(X_test_scaled), args.max_steps)
    for i in range(n_steps):
        sample = X_test_scaled[i]
        t_s = float(i * 0.002)
        out = engine.update(sample, timestamp_s=t_s)
        if out.filtered_alarm == 1 or i % 10 == 0:
            print(f"Step {i:4d} | t={t_s:6.3f}s | Score={out.score:7.3f} (tau={thresh:.3f}) | "
                  f"State={out.operational_state.value:8s} | Latency={out.computational_latency_ms * 1000.0:5.1f}us")
    print("=" * 80)
    print(f"Streaming completed across {n_steps} steps.")

    return 0


def cmd_v08_benchmark(args: argparse.Namespace) -> int:
    """Run V0.8 concept drift and safe adaptation benchmark suite."""
    from qism.drift.experiments import DriftExperimentRunner, DriftExperimentSuiteConfig

    cfg = DriftExperimentSuiteConfig(
        output_dir=args.outdir,
        master_seed=args.seed,
        total_steps=args.steps,
    )
    runner = DriftExperimentRunner(cfg)
    res = runner.run_all()

    import pandas as pd
    summary_df = pd.read_csv(res["summary_csv"])

    print("\n" + "=" * 105)
    print("QISM V0.8 Concept Drift, Distribution Shift & Safe Adaptation Benchmark Summary")
    print("=" * 105)
    cols_to_print = ["experiment_id", "scenario_type", "adaptation_policy", "f1", "recall", "fpr", "drift_detected", "adaptations_accepted", "rollbacks_executed", "mean_latency_ms"]
    cols_available = [c for c in cols_to_print if c in summary_df.columns]
    print(summary_df[cols_available].to_string(index=False))
    print("=" * 105)
    print(f"Results, manifest, and publication figures serialized to: {runner.output_dir}")
    print("=" * 105)
    return 0


def cmd_v08_stream(args: argparse.Namespace) -> int:
    """Run causal adaptive streaming inference on synthetic/physics drift stream."""
    import numpy as np
    import pandas as pd
    from qism.drift.scenarios import DriftScenarioGenerator, DriftScenarioType, ScenarioSpec
    from qism.drift.adaptation import AdaptationConfig, AdaptationPolicy, AdaptationStateMachine
    from qism.drift.detectors import DriftDetectorConfig, StatisticalDriftDetector
    from qism.drift.reference import ReferenceManager, ReferenceStrategy
    from qism.drift.registry import ModelRegistry
    from qism.drift.streaming import AdaptiveStreamingEngine
    from qism.ml.preprocessing.scalers import PreprocessorPipeline
    from qism.temporal.alarm import AlarmConfig, TemporalAlarmStateMachine
    from qism.temporal.models import RollingZScoreDetector

    scen_type = DriftScenarioType(args.scenario)
    spec = ScenarioSpec(scenario_type=scen_type, total_steps=args.steps, seed=args.seed)
    df_stream, feature_cols = DriftScenarioGenerator(spec).generate_stream()

    X_baseline = df_stream[feature_cols].iloc[:40].values
    prep = PreprocessorPipeline(scaler="robust")
    X_base_scaled = prep.fit_transform(pd.DataFrame(X_baseline, columns=feature_cols))

    det = RollingZScoreDetector(window_length=5)
    det.fit(X_base_scaled)
    det.set_threshold(args.alarm_threshold)

    alarm_sm = TemporalAlarmStateMachine(AlarmConfig(tau_high=args.alarm_threshold, persistence_k=2, persistence_n=3))
    drift_det = StatisticalDriftDetector(DriftDetectorConfig(threshold=args.drift_threshold, feature_names=feature_cols))
    ref_mgr = ReferenceManager(strategy=ReferenceStrategy(args.reference_strategy), initial_data=X_base_scaled, feature_names=feature_cols)
    reg = ModelRegistry()
    reg.register_initial_model(det, args.alarm_threshold, ref_mgr.current_version_id)

    def model_builder(cand_data: np.ndarray):
        new_m = RollingZScoreDetector(window_length=5)
        new_m.fit(cand_data)
        new_m.set_threshold(args.alarm_threshold)
        return new_m

    adapt_sm = AdaptationStateMachine(
        AdaptationConfig(policy=AdaptationPolicy(args.policy), candidate_buffer_size=20, validation_window=15),
        reg, ref_mgr, args.alarm_threshold, model_builder=model_builder
    )

    engine = AdaptiveStreamingEngine(
        active_model=det,
        preprocessor=prep,
        alarm_machine=alarm_sm,
        drift_detector=drift_det,
        reference_manager=ref_mgr,
        adaptation_machine=adapt_sm,
        registry=reg,
        history_window_length=5,
        drift_evaluation_window=20,
        feature_names=feature_cols,
    )

    print("\n" + "=" * 110)
    print(f"QISM V0.8 Adaptive Telemetry Monitoring Active | Scenario: {scen_type.value} | Policy: {args.policy}")
    print("=" * 110)

    n_steps = min(len(df_stream), args.max_steps)
    for i in range(n_steps):
        row = df_stream[feature_cols].iloc[i].values
        t_s = float(df_stream["timestamp"].iloc[i])
        out = engine.update(row, timestamp_s=t_s)
        if out.filtered_alarm == 1 or out.is_drift or i % 10 == 0:
            print(f"Step {i:3d} | t={t_s:5.3f}s | Anom={out.anomaly_score:6.2f} (tau={out.anomaly_threshold:.2f}) | "
                  f"Drift={out.drift_score:5.2f} ({'YES' if out.is_drift else ' no'}) | "
                  f"AdaptState={out.adaptation_state.value:17s} | Model={out.active_model_version:9s} | "
                  f"Latency={out.computational_latency_ms * 1000.0:5.1f}us")

    print("=" * 110)
    print(f"Streaming replay completed across {n_steps} steps.")
    return 0


def cmd_v09_benchmark(args: argparse.Namespace) -> int:
    """Run V0.9 integrated security monitoring benchmark suite across all 12 scenarios."""
    from pathlib import Path
    from qism.monitor.experiments import run_all_v09_experiments

    outdir = Path(args.outdir)
    manifest = run_all_v09_experiments(output_dir=outdir)
    print(f"\nV0.9 Benchmark complete. Executed {manifest['total_experiments']} experiments.")
    print(f"Artifacts saved to: {outdir.resolve()}")
    return 0


def cmd_v09_monitor(args: argparse.Namespace) -> int:
    """Run live/replay integrated operational security monitoring with incident tracking."""
    from qism.drift.scenarios import DriftScenarioGenerator, DriftScenarioType, ScenarioSpec
    from qism.monitor.alerts import ConsoleNotifier
    from qism.monitor.orchestrator import MonitorConfig, QISMMonitor

    scen_type = DriftScenarioType(args.scenario)
    spec = ScenarioSpec(scenario_type=scen_type, total_steps=args.steps, seed=args.seed)
    df_stream, feature_cols = DriftScenarioGenerator(spec).generate_stream()

    cfg = MonitorConfig(run_id=f"cli_{args.scenario}", enable_adaptation=(args.policy == "safe"))
    notifier = ConsoleNotifier() if args.verbose else None
    notifiers = [notifier] if notifier else []
    monitor = QISMMonitor(cfg, feature_names=feature_cols, notifiers=notifiers)

    print("\n" + "=" * 115)
    print(f"QISM V0.9 Integrated Security Monitor | Scenario: {scen_type.value} | Steps: {min(len(df_stream), args.max_steps)}")
    print(f"Pipeline: Telemetry -> Signal Fusion -> Alarms -> Incidents -> Severity -> Alerts")
    print("=" * 115)

    n_steps = min(len(df_stream), args.max_steps)
    ingestor = monitor.ingestor
    records = list(ingestor.stream_from_dataframe(df_stream.iloc[:n_steps], feature_cols=feature_cols))

    for idx, rec in enumerate(records):
        out = monitor.update(rec)
        sev_str = out.current_severity.value.upper()
        inc_str = out.active_incidents[0].incident_id if out.active_incidents else "None"
        alt_str = out.new_alert.alert_id if out.new_alert else "-"

        if out.is_anomaly or out.is_drift or out.evidence_score >= 0.40 or idx % 10 == 0:
            print(
                f"Step {idx:3d} | t={out.timestamp_s:5.3f}s | "
                f"Anom={out.anomaly_score:5.2f} | Drift={out.drift_score:4.2f} | "
                f"Evidence={out.evidence_score:4.2f} | Alarms={len(out.active_alarms)} | "
                f"Inc={inc_str:8s} | Sev={sev_str:8s} | Alert={alt_str:8s} | "
                f"Latency={out.processing_latency_ms:4.2f}ms"
            )

    all_incs = monitor.incident_manager.finalize(final_timestamp_s=records[-1].timestamp_s if records else 0.0)
    print("=" * 115)
    print(f"Replay complete. Total Incidents: {len(all_incs)} | Total Raw Alarms: {monitor.incident_manager.total_raw_alarms} | Compression: {monitor.incident_manager.compression_ratio:.1f}x")
    return 0


def cmd_v10_benchmark(args: argparse.Namespace) -> int:
    """Run V1.0 master benchmark suite across multiple seeds and generate research release artifacts."""
    from pathlib import Path
    from qism.monitor.v10_benchmark import run_v10_benchmarks

    outdir = Path(args.outdir)
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    manifest = run_v10_benchmarks(output_dir=outdir, seeds=seed_list)
    print(f"\nV1.0 Master Benchmark complete. Total scenarios: {manifest['total_scenarios']} | Seeds tested: {seed_list}")
    print(f"Release artifacts saved to: {outdir.resolve()}")
    return 0


def cmd_v10_demo(args: argparse.Namespace) -> int:
    """Run canonical single-command V1.0 end-to-end security monitoring demonstration."""
    from qism.monitor.v10_benchmark import run_v10_demo

    return run_v10_demo(
        scenario=args.scenario,
        output_dir=args.outdir,
        seed=args.seed,
        verbose=args.verbose,
    )



def main(argv: list[str] | None = None) -> int:
    verify_python_version()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s")

    parser = argparse.ArgumentParser(prog="qism", description="QISM V1.0 CLI: Quantum Infrastructure Security Monitor")
    parser.add_argument("--version", action="version", version=f"QISM {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)


    # generate-scenarios
    p_scen = subparsers.add_parser("generate-scenarios", help="Generate and preview scenarios")
    p_scen.add_argument("config", help="YAML configuration file path")
    p_scen.set_defaults(func=cmd_generate_scenarios)

    # generate-dataset
    p_data = subparsers.add_parser("generate-dataset", help="Generate complete sharded dataset")
    p_data.add_argument("config", help="YAML configuration file path")
    p_data.add_argument("--workers", type=int, default=None, help="Override worker process count")
    p_data.add_argument("--resume", action="store_true", help="Resume interrupted dataset generation")
    p_data.add_argument("--fresh", action="store_true", help="Wipe existing dataset directory and restart")
    p_data.set_defaults(func=cmd_generate_dataset)

    # resume
    p_res = subparsers.add_parser("resume", help="Resume interrupted dataset generation")
    p_res.add_argument("config", help="YAML configuration file path")
    p_res.add_argument("--workers", type=int, default=None, help="Override worker process count")
    p_res.set_defaults(func=cmd_resume)

    # validate-dataset
    p_val = subparsers.add_parser("validate-dataset", help="Validate dataset schema and check leakage")
    p_val.add_argument("paths", nargs="+", help="Dataset directories to validate")
    p_val.add_argument("--check-split", action="store_true", help="Check scenario-level split leakage")
    p_val.set_defaults(func=cmd_validate_dataset)

    # benchmark
    p_bm = subparsers.add_parser("benchmark", help="Run serial vs parallel throughput benchmark")
    p_bm.add_argument("config", help="YAML configuration file path")
    p_bm.add_argument("--workers", type=int, default=None, help="Workers for parallel execution")
    p_bm.set_defaults(func=cmd_benchmark)

    # train
    p_tr = subparsers.add_parser("train", help="Train anomaly model and calibrate threshold")
    p_tr.add_argument("config", help="Experiment YAML configuration path")
    p_tr.set_defaults(func=cmd_train)

    # evaluate
    p_ev = subparsers.add_parser("evaluate", help="Evaluate saved model on dataset split")
    p_ev.add_argument("experiment_dir", help="Saved experiment directory containing model and config")
    p_ev.add_argument("--split", default="test", help="Split name to evaluate (test, generalization, normal)")
    p_ev.add_argument("--dataset", default=None, help="Optional override dataset root path")
    p_ev.set_defaults(func=cmd_evaluate)

    # experiment
    p_exp = subparsers.add_parser("experiment", help="Run full anomaly detection experiment")
    p_exp.add_argument("config", help="Experiment YAML configuration path")
    p_exp.set_defaults(func=cmd_experiment)

    # benchmark-ml
    p_bml = subparsers.add_parser("benchmark-ml", help="Run multi-model anomaly detection benchmark")
    p_bml.add_argument("config", help="Experiment YAML configuration path")
    p_bml.set_defaults(func=cmd_benchmark_ml)

    # analyze-results
    p_ar = subparsers.add_parser("analyze-results", help="Analyze and display results from experiments directory")
    p_ar.add_argument("experiments_dir", help="Root directory containing experiment runs")
    p_ar.set_defaults(func=cmd_analyze_results)

    # visualize
    p_vis = subparsers.add_parser("visualize", help="Generate presentation-quality visualizations")
    p_vis.add_argument("--dataset", default="data/v0.5_small/v05_small_dataset", help="Input dataset path")
    p_vis.add_argument("--outdir", default="outputs/visualizations", help="Output directory")
    p_vis.add_argument("--formats", default="png,svg,pdf", help="Image formats (comma-separated)")
    p_vis.add_argument("--dpi", type=int, default=300, help="Target DPI")
    p_vis.set_defaults(func=cmd_visualize)

    # v07-benchmark
    p_v07b = subparsers.add_parser("v07-benchmark", help="Run V0.7 temporal anomaly detection benchmark")
    p_v07b.add_argument("--dataset", default="data/v0.5_small/v05_small_dataset", help="Input dataset path")
    p_v07b.add_argument("--outdir", default="outputs/v07/experiments", help="Output directory")
    p_v07b.add_argument("--seed", type=int, default=42, help="Master random seed")
    p_v07b.add_argument("--window-length", type=int, default=5, help="Temporal window length L")
    p_v07b.add_argument("--epochs", type=int, default=15, help="Epochs for sequence neural models")
    p_v07b.set_defaults(func=cmd_v07_benchmark)

    # v07-stream
    p_v07s = subparsers.add_parser("v07-stream", help="Run causal streaming monitoring on test telemetry")
    p_v07s.add_argument("--dataset", default="data/v0.5_small/v05_small_dataset", help="Input dataset path")
    p_v07s.add_argument("--window-length", type=int, default=5, help="Temporal window length L")
    p_v07s.add_argument("--max-steps", type=int, default=50, help="Max streaming steps to simulate")
    p_v07s.set_defaults(func=cmd_v07_stream)

    # v08-benchmark
    p_v08b = subparsers.add_parser("v08-benchmark", help="Run V0.8 concept drift and safe adaptation benchmark suite")
    p_v08b.add_argument("--outdir", default="outputs/v08/experiments", help="Output directory")
    p_v08b.add_argument("--seed", type=int, default=42, help="Master random seed")
    p_v08b.add_argument("--steps", type=int, default=150, help="Steps per scenario trajectory")
    p_v08b.set_defaults(func=cmd_v08_benchmark)

    # v08-stream
    p_v08s = subparsers.add_parser("v08-stream", help="Run causal adaptive streaming on drift scenario")
    p_v08s.add_argument("--scenario", default="drift_then_attack", help="Scenario type (abrupt_benign, gradual_benign, recurring_benign, drift_then_attack, contamination_stress)")
    p_v08s.add_argument("--policy", default="safe", help="Adaptation policy (fixed, naive, safe)")
    p_v08s.add_argument("--reference-strategy", default="anchored", help="Reference strategy (fixed, sliding, anchored)")
    p_v08s.add_argument("--steps", type=int, default=120, help="Total scenario trajectory steps")
    p_v08s.add_argument("--max-steps", type=int, default=60, help="Max steps to stream")
    p_v08s.add_argument("--seed", type=int, default=42, help="Random seed")
    p_v08s.add_argument("--alarm-threshold", type=float, default=3.0, help="Alarm decision threshold")
    p_v08s.add_argument("--drift-threshold", type=float, default=0.45, help="Drift decision threshold")
    p_v08s.set_defaults(func=cmd_v08_stream)

    # v09-benchmark
    p_v09b = subparsers.add_parser("v09-benchmark", help="Run V0.9 integrated security monitoring benchmark suite")
    p_v09b.add_argument("--outdir", default="outputs/v09/experiments", help="Output directory")
    p_v09b.set_defaults(func=cmd_v09_benchmark)

    # v09-monitor
    p_v09m = subparsers.add_parser("v09-monitor", help="Run V0.9 integrated operational monitoring replay")
    p_v09m.add_argument("--scenario", default="drift_then_attack", help="Scenario type (stationary, abrupt_benign, drift_then_attack, attack_during_drift, ood_generalization)")
    p_v09m.add_argument("--policy", default="safe", help="Adaptation policy (safe, fixed)")
    p_v09m.add_argument("--steps", type=int, default=120, help="Total scenario steps")
    p_v09m.add_argument("--max-steps", type=int, default=120, help="Max steps to monitor")
    p_v09m.add_argument("--seed", type=int, default=42, help="Random seed")
    p_v09m.add_argument("--verbose", action="store_true", help="Print real-time alert notifications")
    p_v09m.set_defaults(func=cmd_v09_monitor)

    # v10-benchmark
    p_v10b = subparsers.add_parser("v10-benchmark", help="Run V1.0 master benchmark suite across multiple seeds")
    p_v10b.add_argument("--outdir", default="outputs/v1_0", help="Root directory for V1.0 release artifacts")
    p_v10b.add_argument("--seeds", default="42,101,2024", help="Comma-separated random seeds (default: 42,101,2024)")
    p_v10b.set_defaults(func=cmd_v10_benchmark)

    # v10-demo
    p_v10d = subparsers.add_parser("v10-demo", help="Run V1.0 canonical end-to-end security monitoring demonstration")
    p_v10d.add_argument("--scenario", default="drift_then_attack", help="Scenario type (drift_then_attack, attack_during_drift, etc.)")
    p_v10d.add_argument("--outdir", default="outputs/v1_0/demo", help="Output directory for demo artifacts")
    p_v10d.add_argument("--seed", type=int, default=42, help="Random seed")
    p_v10d.add_argument("--verbose", action="store_true", default=True, help="Display live step-by-step console alerts")
    p_v10d.set_defaults(func=cmd_v10_demo)

    args = parser.parse_args(argv)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
