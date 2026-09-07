"""Experiment runner executing complete end-to-end benchmark workflows (Requirements 53–56, 146–148, 183–185)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qism.ml.data.loader import MLDatasetLoader
from qism.ml.evaluation.runner import EvaluationRunner
from qism.ml.experiments.config import ExperimentConfig
from qism.ml.experiments.diagnostics import (
    plot_anomaly_timeline,
    plot_evaluation_curves,
    plot_score_distributions,
    run_leakage_sanity_experiments,
)
from qism.ml.features.extractor import FeatureExtractor
from qism.ml.models.registry import build_model
from qism.ml.preprocessing.scalers import PreprocessorPipeline
from qism.ml.thresholding.calibration import ThresholdCalibrator


class ExperimentRunner:
    """Coordinates data loading, preprocessing, model fitting, threshold calibration, evaluation, and reporting."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.exp_dir = Path(config.output_dir) / config.experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.exp_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        """Execute the end-to-end experiment."""
        start_time = datetime.now(timezone.utc)

        # 1. Load dataset with scenario-level split isolation
        loader = MLDatasetLoader(
            dataset_root=self.config.dataset_root,
            master_seed=self.config.master_seed,
            train_val_split_ratio=self.config.train_val_split_ratio,
        )

        df_train_raw = loader.load_split_dataframe("train")
        df_val_raw = loader.load_split_dataframe("val")
        df_test_raw = loader.load_split_dataframe("test")
        df_gen_raw = loader.load_split_dataframe("generalization")

        # 2. Extract features
        extractor = FeatureExtractor(
            feature_set=self.config.feature_set,
            add_temporal_features=self.config.add_temporal_features,
            rolling_window_length=self.config.rolling_window_length,
        )

        X_train_df, _ = extractor.extract(df_train_raw)
        X_val_df, y_val_df = extractor.extract(df_val_raw)

        # 3. Fit preprocessing strictly on normal training data
        preprocessor = PreprocessorPipeline(
            scaler=self.config.scaler,
            impute_strategy=self.config.impute_strategy,
            clip_sigma=self.config.clip_sigma,
        )
        X_train_scaled = preprocessor.fit_transform(X_train_df)
        X_val_scaled = preprocessor.transform(X_val_df)

        # 4. Fit model
        model = build_model(self.config.model_type, **self.config.hyperparameters)
        model.fit(X_train_scaled)

        # 5. Calibrate threshold on normal validation data
        val_scores = model.score(X_val_scaled)
        calibrator = ThresholdCalibrator(target_fpr=self.config.target_fpr)
        calib_res = calibrator.calibrate(val_scores)
        threshold = calib_res["threshold"]
        model.set_threshold(threshold)

        # 6. Evaluate on In-Distribution (IID) Test Split
        eval_runner = EvaluationRunner(
            model=model,
            preprocessor=preprocessor,
            extractor=extractor,
            threshold=threshold,
            consecutive_k=self.config.consecutive_k,
        )
        iid_results = eval_runner.evaluate(df_test_raw, split_name="test_iid")

        # 7. Evaluate on Out-of-Distribution (OOD) Generalization Split
        ood_results = eval_runner.evaluate(df_gen_raw, split_name="generalization_ood")

        # 8. Calculate OOD Generalization Gap (Delta M = M_IID - M_OOD)
        def _get_metric(res: dict[str, Any], section: str, key: str) -> float:
            return float(res.get(section, {}).get(key, 0.0))

        iid_rec = _get_metric(iid_results, "window_metrics", "recall")
        ood_rec = _get_metric(ood_results, "window_metrics", "recall")
        iid_fpr = _get_metric(iid_results, "window_metrics", "fpr")
        ood_fpr = _get_metric(ood_results, "window_metrics", "fpr")
        iid_f1 = _get_metric(iid_results, "window_metrics", "f1")
        ood_f1 = _get_metric(ood_results, "window_metrics", "f1")
        iid_auroc = _get_metric(iid_results, "curve_metrics", "auroc")
        ood_auroc = _get_metric(ood_results, "curve_metrics", "auroc")

        generalization_gap = {
            "recall_gap": iid_rec - ood_rec,
            "fpr_gap": ood_fpr - iid_fpr,  # Increase in FPR is degradation
            "f1_gap": iid_f1 - ood_f1,
            "auroc_gap": iid_auroc - ood_auroc,
        }

        # 9. Run Sanity Experiments
        sanity_results = run_leakage_sanity_experiments(
            model=model,
            preprocessor=preprocessor,
            extractor=extractor,
            df_test=df_test_raw,
            seed=self.config.master_seed,
        )

        # 10. Generate and save diagnostic figures
        plot_evaluation_curves(iid_results, self.plots_dir)

        # Score distributions
        if not df_test_raw.empty:
            X_test_df, y_test_df = extractor.extract(df_test_raw)
            X_test_scaled = preprocessor.transform(X_test_df)
            test_scores = model.score(X_test_scaled)
            atk_mask = y_test_df["attack_active"].to_numpy(dtype=bool)
            norm_scores = test_scores[~atk_mask]
            atk_scores = test_scores[atk_mask]

            plot_score_distributions(
                normal_scores=norm_scores,
                attack_scores=atk_scores,
                threshold=threshold,
                output_path=self.plots_dir / "score_distributions.png",
            )

            # Representative Timeline
            plot_anomaly_timeline(
                timestamps=range(len(test_scores)),
                scores=test_scores,
                threshold=threshold,
                attack_active=atk_mask,
                alarms=(test_scores > threshold).astype(int),
                output_path=self.plots_dir / "anomaly_timeline.png",
            )

        # 11. Persist model and preprocessor artifacts
        model_save_path = self.exp_dir / "model.bin"
        model.save(model_save_path)
        preprocessor.save(self.exp_dir / "preprocessor.json")
        self.config.to_yaml(self.exp_dir / "config.yaml")

        # 12. Create Experiment Manifest
        end_time = datetime.now(timezone.utc)
        manifest = {
            "experiment_id": self.config.experiment_id,
            "version": self.config.version,
            "model_type": self.config.model_type,
            "feature_set": self.config.feature_set,
            "scaler": self.config.scaler,
            "target_fpr": self.config.target_fpr,
            "calibrated_threshold": threshold,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "wall_time_s": (end_time - start_time).total_seconds(),
            "master_seed": self.config.master_seed,
            "status": "COMPLETED",
        }
        with open(self.exp_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # 13. Save consolidated results
        consolidated = {
            "manifest": manifest,
            "calibration": calib_res,
            "iid_test": iid_results,
            "ood_generalization": ood_results,
            "generalization_gap": generalization_gap,
            "sanity_experiments": sanity_results,
        }
        with open(self.exp_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2, default=str)

        # 14. Save machine-readable summary CSV
        summary_row = {
            "experiment_id": self.config.experiment_id,
            "model_type": self.config.model_type,
            "feature_set": self.config.feature_set,
            "threshold": threshold,
            "iid_precision": iid_results.get("window_metrics", {}).get("precision", 0.0),
            "iid_recall": iid_rec,
            "iid_f1": iid_f1,
            "iid_fpr": iid_fpr,
            "iid_auroc": iid_auroc,
            "iid_auprc": iid_results.get("curve_metrics", {}).get("auprc", 0.0),
            "ood_precision": ood_results.get("window_metrics", {}).get("precision", 0.0),
            "ood_recall": ood_rec,
            "ood_f1": ood_f1,
            "ood_fpr": ood_fpr,
            "ood_auroc": ood_auroc,
            "ood_auprc": ood_results.get("curve_metrics", {}).get("auprc", 0.0),
            "recall_gap": generalization_gap["recall_gap"],
            "fpr_gap": generalization_gap["fpr_gap"],
            "shuffled_auroc": sanity_results.get("shuffled_labels", {}).get("auroc", 0.5),
            "random_feature_auroc": sanity_results.get("random_features", {}).get("auroc", 0.5),
        }
        pd.DataFrame([summary_row]).to_csv(self.exp_dir / "summary.csv", index=False)

        return consolidated
