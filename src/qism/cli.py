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


def main(argv: list[str] | None = None) -> int:
    verify_python_version()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s")

    parser = argparse.ArgumentParser(prog="qism", description="QISM Simulator V0.6 CLI")
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
