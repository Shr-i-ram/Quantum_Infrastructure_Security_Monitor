"""Evaluation runner coordinating window and instance metrics across splits (Requirements 92, 183–191, 203–208)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qism.ml.evaluation.instance import AttackInstanceEvaluator
from qism.ml.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_continuous_curve_metrics,
)
from qism.ml.features.extractor import FeatureExtractor
from qism.ml.models.base import AnomalyDetector
from qism.ml.preprocessing.scalers import PreprocessorPipeline
from qism.ml.thresholding.alarm import AlarmStateMachine


class EvaluationRunner:
    """Evaluates an anomaly detector across in-distribution and out-of-distribution splits."""

    def __init__(
        self,
        model: AnomalyDetector,
        preprocessor: PreprocessorPipeline,
        extractor: FeatureExtractor,
        threshold: float,
        consecutive_k: int = 2,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.threshold = threshold
        self.consecutive_k = consecutive_k
        self.instance_evaluator = AttackInstanceEvaluator()

    def evaluate(
        self,
        df_raw: pd.DataFrame,
        split_name: str = "test",
    ) -> dict[str, Any]:
        """Run complete evaluation over raw telemetry DataFrame for a given split."""
        if df_raw.empty:
            return {"split": split_name, "error": "Empty dataset"}

        # 1. Feature extraction
        X_df, y_df = self.extractor.extract(df_raw)

        # 2. Preprocessing
        X_scaled = self.preprocessor.transform(X_df)

        # 3. Continuous scoring and prediction
        scores = self.model.score(X_scaled)
        y_true = y_df["attack_active"].to_numpy(dtype=int)
        y_pred = (scores > self.threshold).astype(int)

        # 4. State machine operational alarms
        alarm_machine = AlarmStateMachine(
            tau_high=self.threshold,
            tau_low=self.threshold * 0.90,
            consecutive_k=self.consecutive_k,
        )
        states, operational_alarms = alarm_machine.process_sequence(scores)

        # 5. Window-level metrics
        classif_metrics = calculate_classification_metrics(y_true, y_pred)
        curve_metrics = calculate_continuous_curve_metrics(y_true, scores)

        # Operational alarm classification metrics
        operational_metrics = calculate_classification_metrics(y_true, operational_alarms)

        # 6. Attack-instance metrics
        df_for_inst = pd.DataFrame({
            "scenario_id": y_df.get("scenario_id", "default"),
            "run_id": y_df.get("run_id", "default"),
            "window_id": y_df.get("window_id", range(len(y_df))),
            "window_start_pulse": y_df.get("window_start_pulse", np.arange(len(y_df)) * 1000),
            "attack_active": y_df.get("attack_active", False),
            "alarm": operational_alarms,
            "attack_type": y_df.get("attack_type", "none"),
            "attack_severity": y_df.get("attack_severity", 0.0),
        })
        inst_metrics = self.instance_evaluator.evaluate_instances(df_for_inst)

        # 7. Attack-type breakdown
        type_breakdown = {}
        if "attack_type" in y_df.columns:
            for atk_type, grp in y_df.groupby("attack_type"):
                if atk_type == "none":
                    continue
                indices = grp.index
                type_y_true = y_true[indices]
                type_y_pred = y_pred[indices]
                type_breakdown[atk_type] = {
                    "count": len(indices),
                    "recall": float(np.mean(type_y_pred == 1)) if len(type_y_pred) else 0.0,
                    "mean_score": float(np.mean(scores[indices])),
                }

        return {
            "split": split_name,
            "threshold": self.threshold,
            "n_samples": len(X_df),
            "window_metrics": classif_metrics,
            "curve_metrics": curve_metrics,
            "operational_metrics": operational_metrics,
            "instance_metrics": inst_metrics,
            "attack_type_breakdown": type_breakdown,
            "scores_summary": {
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "median": float(np.median(scores)),
            },
        }
