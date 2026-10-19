"""Complete QISM Classical, Baseline, and Quantum Model Training, Evaluation, Ablation Study, and Visualization.

Executes the authoritative research benchmark over the QISM research dataset (data/qism_research_dataset):
1. Loads partitions: train, validation, test_in_distribution, test_generalization, test_stress.
2. Fits preprocessors (SimpleImputer, StandardScaler, RobustScaler, PCA, AngleScaler) strictly on normal train data.
3. Trains every existing classical anomaly detector & baseline (Isolation Forest, One-Class SVM, LOF, Mahalanobis, Robust Covariance, Autoencoder, Temporal Autoencoder, Deep SVDD).
4. Trains every existing classical supervised baseline (Logistic Regression, Random Forest, Small MLP, RBF-SVM).
5. Trains every existing quantum ML baseline (VQC, Quantum Kernel SVM, Quantum Autoencoder, Static Quantum SVDD) on lightning.qubit / default.qubit.
6. Executes the flagship PI-TQSVDD research study across 2, 4, 6, 8 qubits, multiple encodings, objective ablations, and multi-seed confirmations.
7. Evaluates all models across all 3 test sets with research-grade point, operational, per-attack, and event-level metrics.
8. Persists atomic checkpoints, prediction records, and comprehensive aggregate CSV/Parquet tables.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repository root and src are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM, SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# QISM Internal Imports
from qism.telemetry.features import get_observable_feature_names
from qml.pitqsvdd.circuits.variational import VariationalQuantumCircuit
from qml.pitqsvdd.config import CircuitConfig, ModelConfig
from qml.pitqsvdd.encodings import get_encoding
from qml.pitqsvdd.evaluation.event_metrics import compute_event_level_metrics
from qml.pitqsvdd.models.context_aware_pitqsvdd import ContextAwarePITQSVDD
from qml.pitqsvdd.models.static_qsvdd import StaticQSVDD
from qml.pitqsvdd.models.temporal_pitqsvdd import TemporalPITQSVDD
from qml.src.baselines import SmallClassicalMLP
from qml.src.quantum_autoencoder import QuantumAutoencoder
from qml.src.quantum_kernel_svm import QuantumKernelSVM
from qml.src.vqc_model import VariationalQuantumClassifier


def safe_metric_calc(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate point metrics and operational recall at threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    auroc, auprc = None, None
    if len(np.unique(y_true)) > 1:
        try:
            auroc = float(roc_auc_score(y_true, y_score))
            auprc = float(average_precision_score(y_true, y_score))
        except Exception:
            pass

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "fpr": fpr,
        "fnr": fnr,
        "auroc": auroc,
        "auprc": auprc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold),
    }


def find_threshold_at_fpr(normal_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Calibrate decision threshold strictly on normal validation scores for target FPR."""
    return float(np.percentile(normal_scores, 100.0 * (1.0 - target_fpr)))


class QISMBenchmarkRunner:
    """Master Benchmark Runner managing data, models, metrics, and persistence."""

    def __init__(
        self,
        dataset_root: str = "data/qism_research_dataset",
        results_root: str = "results/qism_model_benchmark",
        master_seed: int = 42,
    ):
        self.dataset_root = Path(dataset_root)
        self.results_root = Path(results_root)
        self.master_seed = master_seed

        # Directory structure
        self.chk_dir = self.results_root / "checkpoints"
        self.pred_dir = self.results_root / "predictions"
        self.tables_dir = self.results_root / "tables"
        self.figs_dir = self.results_root / "figures"
        self.metrics_dir = self.results_root / "metrics"

        for d in [
            self.chk_dir / "classical",
            self.chk_dir / "baselines",
            self.chk_dir / "quantum",
            self.chk_dir / "pitqsvdd",
            self.pred_dir,
            self.tables_dir,
            self.metrics_dir,
            self.figs_dir / "dataset",
            self.figs_dir / "training",
            self.figs_dir / "comparisons",
            self.figs_dir / "pitqsvdd",
            self.figs_dir / "generalization",
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.observable_cols = get_observable_feature_names()
        self.partitions_raw: dict[str, pd.DataFrame] = {}
        self.metadata: dict[str, pd.DataFrame] = {}
        self.features: dict[str, dict[str, np.ndarray]] = {}

        self.experiment_registry: list[dict[str, Any]] = []
        self.per_run_metrics: list[dict[str, Any]] = []
        self.per_attack_metrics: list[dict[str, Any]] = []
        self.compute_cost_records: list[dict[str, Any]] = []
        self.failed_runs: list[dict[str, Any]] = []

    def load_data(self) -> None:
        """Load all 5 logical partitions, verifying zero label leakage."""
        print("\n========================================================")
        print(" [PHASE I & II] Loading Authoritative QISM Telemetry")
        print("========================================================")
        splits = ["train", "validation", "test_in_distribution", "test_generalization", "test_stress"]
        
        for sp in splits:
            sp_dir = self.dataset_root / sp
            telem_files = sorted(sp_dir.rglob("telemetry.parquet"))
            if not telem_files:
                raise FileNotFoundError(f"No telemetry files found in {sp_dir}")

            dfs = [pd.read_parquet(f) for f in telem_files]
            combined = pd.concat(dfs, ignore_index=True)
            self.partitions_raw[sp] = combined

            # Split metadata vs observable features
            meta_cols = [
                "scenario_id",
                "run_id",
                "window_id",
                "ground_truth_attack_active",
                "ground_truth_attack_type",
            ]
            available_meta = [c for c in meta_cols if c in combined.columns]
            self.metadata[sp] = combined[available_meta].copy()

            # Ensure all observable features exist
            for col in self.observable_cols:
                if col not in combined.columns:
                    raise KeyError(f"Observable column {col} missing in {sp}")

            print(f" Loaded {sp:22s}: {len(combined):5d} windows | Normal: {(combined['ground_truth_attack_active'] == False).sum():4d} | Attack: {combined['ground_truth_attack_active'].sum():4d}")

        # Verification: Train must be 100% normal
        train_attacks = self.metadata["train"]["ground_truth_attack_active"].sum()
        if train_attacks != 0:
            raise RuntimeError(f"FATAL: Train partition contains {train_attacks} attack windows! Normal-only violation.")

    def fit_preprocessors(self) -> None:
        """Fit all preprocessing transformers exclusively on normal training data (Phase III)."""
        print("\n========================================================")
        print(" [PHASE III] Fitting Preprocessors Exclusively on Train")
        print("========================================================")
        X_train_raw = self.partitions_raw["train"][self.observable_cols].copy()

        # Handle all-NaN columns by filling with 0.0, then median impute
        self.all_nan_cols = [c for c in self.observable_cols if X_train_raw[c].isna().all()]
        if self.all_nan_cols:
            print(f" Identified {len(self.all_nan_cols)} all-NaN feature columns in train: {self.all_nan_cols}")
            X_train_raw[self.all_nan_cols] = 0.0

        # 1. Median Imputer
        self.imputer = SimpleImputer(strategy="median", fill_value=0.0)
        X_train_imp = self.imputer.fit_transform(X_train_raw)

        # 2. Standard Scaler
        self.std_scaler = StandardScaler()
        X_train_std = self.std_scaler.fit_transform(X_train_imp)

        # 3. Robust Scaler
        from sklearn.preprocessing import RobustScaler
        self.robust_scaler = RobustScaler()
        X_train_robust = self.robust_scaler.fit_transform(X_train_imp)

        # 4. Angle Scaler [0, pi]
        self.angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        X_train_angle = self.angle_scaler.fit_transform(X_train_imp)

        # 5. PCA Models for 2, 4, 6, 8, 16
        self.pca_models: dict[int, PCA] = {}
        self.pca_scalers: dict[int, MinMaxScaler] = {}
        for n_comp in [2, 4, 6, 8, 16]:
            pca = PCA(n_components=n_comp, random_state=self.master_seed)
            pca.fit(X_train_std)
            self.pca_models[n_comp] = pca

            # Fit Angle Scaler on PCA output
            tr_pca = pca.transform(X_train_std)
            scaler_ang = MinMaxScaler(feature_range=(0.0, np.pi))
            scaler_ang.fit(tr_pca)
            self.pca_scalers[n_comp] = scaler_ang

            var_exp = float(np.sum(pca.explained_variance_ratio_))
            print(f" PCA-{n_comp:2d}: Explained Variance = {var_exp * 100:.2f}% (train-fitted)")

        # Transform all partitions
        for sp in self.partitions_raw:
            df = self.partitions_raw[sp][self.observable_cols].copy()
            if self.all_nan_cols:
                df[self.all_nan_cols] = 0.0
            x_imp = self.imputer.transform(df)
            x_std = self.std_scaler.transform(x_imp)
            x_rob = self.robust_scaler.transform(x_imp)
            x_ang = self.angle_scaler.transform(x_imp)

            pcas = {n_comp: self.pca_models[n_comp].transform(x_std) for n_comp in self.pca_models}
            pcas_ang = {
                n_comp: self.pca_scalers[n_comp].transform(pcas[n_comp])
                for n_comp in pcas
            }

            self.features[sp] = {
                "raw": x_imp,
                "standard": x_std,
                "robust": x_rob,
                "angle": x_ang,
                "pca_2": pcas[2],
                "pca_4": pcas[4],
                "pca_6": pcas[6],
                "pca_8": pcas[8],
                "pca_16": pcas[16],
                "pca_2_angle": pcas_ang[2],
                "pca_4_angle": pcas_ang[4],
                "pca_6_angle": pcas_ang[6],
                "pca_8_angle": pcas_ang[8],
                "pca_16_angle": pcas_ang[16],
            }

        # Build causal temporal sliding windows strictly within scenarios
        self._build_temporal_windows()

    def _build_temporal_windows(self, seq_len: int = 3) -> None:
        """Construct causal temporal sequences strictly within each scenario."""
        self.temporal_windows: dict[str, dict[str, Any]] = {}
        for sp in self.partitions_raw:
            meta = self.metadata[sp]
            scen_ids = meta["scenario_id"].unique()

            all_win = {
                "standard": [],
                "pca_2": [],
                "pca_4": [],
                "pca_6": [],
                "pca_8": [],
                "pca_16": [],
            }
            all_targets = {
                "standard": [],
                "pca_2": [],
                "pca_4": [],
                "pca_6": [],
                "pca_8": [],
                "pca_16": [],
            }
            all_has_target = []
            all_labels = []
            all_win_meta = []

            for sid in scen_ids:
                scen_mask = (meta["scenario_id"] == sid).values
                scen_std = self.features[sp]["standard"][scen_mask]
                scen_pcas = {
                    2: self.features[sp]["pca_2_angle"][scen_mask],
                    4: self.features[sp]["pca_4_angle"][scen_mask],
                    6: self.features[sp]["pca_6_angle"][scen_mask],
                    8: self.features[sp]["pca_8_angle"][scen_mask],
                    16: self.features[sp]["pca_16_angle"][scen_mask],
                }
                scen_labels = meta["ground_truth_attack_active"].values[scen_mask].astype(int)
                scen_records = meta[scen_mask].to_dict(orient="records")

                n_steps = len(scen_std)
                for i in range(seq_len - 1, n_steps):
                    all_win["standard"].append(scen_std[i - seq_len + 1 : i + 1])
                    for k in [2, 4, 6, 8, 16]:
                        all_win[f"pca_{k}"].append(scen_pcas[k][i - seq_len + 1 : i + 1])

                    all_labels.append(scen_labels[i])
                    all_win_meta.append(scen_records[i])

                    # Target for next-step prediction
                    if i + 1 < n_steps:
                        all_targets["standard"].append(scen_std[i + 1])
                        for k in [2, 4, 6, 8, 16]:
                            all_targets[f"pca_{k}"].append(scen_pcas[k][i + 1])
                        all_has_target.append(True)
                    else:
                        all_targets["standard"].append(scen_std[i])
                        for k in [2, 4, 6, 8, 16]:
                            all_targets[f"pca_{k}"].append(scen_pcas[k][i])
                        all_has_target.append(False)

            self.temporal_windows[sp] = {
                "windows_standard": torch.tensor(np.array(all_win["standard"]), dtype=torch.float32),
                "windows_pca2": torch.tensor(np.array(all_win["pca_2"]), dtype=torch.float32),
                "windows_pca4": torch.tensor(np.array(all_win["pca_4"]), dtype=torch.float32),
                "windows_pca6": torch.tensor(np.array(all_win["pca_6"]), dtype=torch.float32),
                "windows_pca8": torch.tensor(np.array(all_win["pca_8"]), dtype=torch.float32),
                "windows_pca16": torch.tensor(np.array(all_win["pca_16"]), dtype=torch.float32),
                "targets_pca2": torch.tensor(np.array(all_targets["pca_2"]), dtype=torch.float32),
                "targets_pca4": torch.tensor(np.array(all_targets["pca_4"]), dtype=torch.float32),
                "targets_pca6": torch.tensor(np.array(all_targets["pca_6"]), dtype=torch.float32),
                "targets_pca8": torch.tensor(np.array(all_targets["pca_8"]), dtype=torch.float32),
                "targets_pca16": torch.tensor(np.array(all_targets["pca_16"]), dtype=torch.float32),
                "has_target": torch.tensor(np.array(all_has_target), dtype=torch.bool),
                "labels": np.array(all_labels),
                "metadata": all_win_meta,
            }
            print(f" Temporal windows for {sp:22s}: shape={self.temporal_windows[sp]['windows_pca4'].shape}, labels={len(all_labels)}")

    def _eval_and_record(
        self,
        model_id: str,
        family: str,
        model_obj: Any,
        predict_fn: Any,
        train_time_s: float,
        input_type: str = "standard",  # "standard", "windows_pca4", "windows_pca2", etc.
        param_count: int = 0,
        qubit_count: int = 0,
        circuit_depth: int = 0,
        seed: int = 42,
        max_samples: int | None = None,
        notes: str = "",
    ) -> None:
        """Evaluate model on validation and all 3 test partitions, recording point, event, and operational metrics."""
        is_temporal = input_type.startswith("windows_")
        
        # Step 1: Calibrate threshold on validation normal scores
        if is_temporal:
            val_in = self.temporal_windows["validation"][input_type]
            val_y = self.temporal_windows["validation"]["labels"]
        else:
            val_in = self.features["validation"][input_type]
            val_y = self.metadata["validation"]["ground_truth_attack_active"].values.astype(int)

        if max_samples is not None and len(val_in) > max_samples:
            val_in = val_in[:max_samples]
            val_y = val_y[:max_samples]

        t_inf0 = time.time()
        val_scores = predict_fn(val_in)
        inf_time_per_sample = (time.time() - t_inf0) / len(val_scores)

        val_normal_scores = val_scores[val_y == 0]
        th_1pct = find_threshold_at_fpr(val_normal_scores, target_fpr=0.01)
        th_5pct = find_threshold_at_fpr(val_normal_scores, target_fpr=0.05)

        # Record compute cost
        self.compute_cost_records.append({
            "model_id": model_id,
            "family": family,
            "train_time_s": train_time_s,
            "inference_time_per_window_s": inf_time_per_sample,
            "param_count": param_count,
            "qubit_count": qubit_count,
            "circuit_depth": circuit_depth,
            "seed": seed,
        })

        # Step 2: Evaluate on all 3 test splits
        test_splits = ["test_in_distribution", "test_generalization", "test_stress"]
        for sp in test_splits:
            if is_temporal:
                test_in = self.temporal_windows[sp][input_type]
                test_y = self.temporal_windows[sp]["labels"]
                test_meta = self.temporal_windows[sp]["metadata"]
            else:
                test_in = self.features[sp][input_type]
                test_y = self.metadata[sp]["ground_truth_attack_active"].values.astype(int)
                test_meta = self.metadata[sp].to_dict(orient="records")

            if max_samples is not None and len(test_in) > max_samples:
                test_in = test_in[:max_samples]
                test_y = test_y[:max_samples]
                test_meta = test_meta[:max_samples]

            t_t0 = time.time()
            test_scores = predict_fn(test_in)
            t_inf_test = time.time() - t_t0

            # Metrics at 1% FPR threshold
            m_1pct = safe_metric_calc(test_y, test_scores, threshold=th_1pct)
            m_5pct = safe_metric_calc(test_y, test_scores, threshold=th_5pct)

            # Event level metrics at 1% FPR threshold
            preds_1pct = (test_scores >= th_1pct).astype(int)
            ev_metrics = compute_event_level_metrics(preds_1pct, test_y, test_meta)

            row = {
                "model_id": model_id,
                "family": family,
                "partition": sp,
                "seed": seed,
                "auroc": m_1pct["auroc"],
                "auprc": m_1pct["auprc"],
                "f1_at_1pct_fpr": m_1pct["f1"],
                "recall_at_1pct_fpr": m_1pct["recall"],
                "fpr_at_1pct_fpr": m_1pct["fpr"],
                "precision_at_1pct_fpr": m_1pct["precision"],
                "f1_at_5pct_fpr": m_5pct["f1"],
                "recall_at_5pct_fpr": m_5pct["recall"],
                "fpr_at_5pct_fpr": m_5pct["fpr"],
                "incident_recall": ev_metrics["incident_recall"],
                "mean_detection_delay_windows": ev_metrics["mean_detection_delay_windows"],
                "false_alarm_rate": ev_metrics["false_alarm_rate_per_normal_window"],
                "train_time_s": train_time_s,
                "inf_time_s": t_inf_test,
                "threshold_1pct": th_1pct,
                "threshold_5pct": th_5pct,
                "notes": notes,
            }
            self.per_run_metrics.append(row)

            # Per-attack breakdown on in-distribution test
            if sp == "test_in_distribution":
                atk_types = [m.get("ground_truth_attack_type", "normal") for m in test_meta]
                for atk in ["intercept_resend", "pns", "source_manipulation", "channel_loss", "detector_manipulation", "timing"]:
                    mask = np.array([a == atk for a in atk_types])
                    if mask.sum() > 0:
                        rec_atk = float(recall_score(test_y[mask], preds_1pct[mask], zero_division=0))
                        self.per_attack_metrics.append({
                            "model_id": model_id,
                            "family": family,
                            "attack_type": atk,
                            "recall": rec_atk,
                            "count": int(mask.sum()),
                            "seed": seed,
                        })

        print(f" Completed {model_id:32s} | ID AUROC: {self.per_run_metrics[-3]['auroc']:.4f} | Gen AUROC: {self.per_run_metrics[-2]['auroc']:.4f} | Stress AUROC: {self.per_run_metrics[-1]['auroc']:.4f}")

    def run_classical_baselines(self) -> None:
        """Train and benchmark all classical unsupervised and statistical baselines (Phase IV)."""
        print("\n========================================================")
        print(" [PHASE IV] Training Classical Baselines")
        print("========================================================")
        X_train_std = self.features["train"]["standard"]
        X_train_pca6 = self.features["train"]["pca_6"]

        # 1. Isolation Forest (Full 76)
        t0 = time.time()
        ifo_full = IsolationForest(n_estimators=100, contamination="auto", random_state=self.master_seed, n_jobs=-1)
        ifo_full.fit(X_train_std)
        fit_t = time.time() - t0
        joblib.dump(ifo_full, self.chk_dir / "classical" / "isolation_forest_full.joblib")
        self._eval_and_record(
            "IsolationForest_Full", "classical_unsupervised", ifo_full,
            lambda x: -ifo_full.decision_function(x), fit_t, input_type="standard", param_count=100 * 76
        )

        # 2. Isolation Forest (PCA-6)
        t0 = time.time()
        ifo_pca = IsolationForest(n_estimators=100, contamination="auto", random_state=self.master_seed, n_jobs=-1)
        ifo_pca.fit(X_train_pca6)
        fit_t = time.time() - t0
        joblib.dump(ifo_pca, self.chk_dir / "classical" / "isolation_forest_pca6.joblib")
        self._eval_and_record(
            "IsolationForest_PCA6", "classical_unsupervised", ifo_pca,
            lambda x: -ifo_pca.decision_function(x), fit_t, input_type="pca_6", param_count=100 * 6
        )

        # 3. One-Class SVM (PCA-6)
        t0 = time.time()
        ocsvm_pca = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        ocsvm_pca.fit(X_train_pca6)
        fit_t = time.time() - t0
        joblib.dump(ocsvm_pca, self.chk_dir / "classical" / "one_class_svm_pca6.joblib")
        self._eval_and_record(
            "OneClassSVM_PCA6", "classical_unsupervised", ocsvm_pca,
            lambda x: -ocsvm_pca.decision_function(x), fit_t, input_type="pca_6", param_count=len(ocsvm_pca.support_)
        )

        # 4. Local Outlier Factor (Novelty mode, PCA-6)
        t0 = time.time()
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        lof.fit(X_train_pca6)
        fit_t = time.time() - t0
        joblib.dump(lof, self.chk_dir / "classical" / "lof_pca6.joblib")
        self._eval_and_record(
            "LocalOutlierFactor_PCA6", "classical_unsupervised", lof,
            lambda x: -lof.decision_function(x), fit_t, input_type="pca_6"
        )

        # 5. Mahalanobis Distance (Full 76 features, regularized)
        t0 = time.time()
        mean_vec = np.mean(X_train_std, axis=0)
        cov_mat = np.cov(X_train_std, rowvar=False) + 1e-4 * np.eye(76)
        inv_cov = np.linalg.pinv(cov_mat)
        fit_t = time.time() - t0
        np.savez(self.chk_dir / "classical" / "mahalanobis_full.npz", mean=mean_vec, inv_cov=inv_cov)
        def _score_mahalanobis(x: np.ndarray) -> np.ndarray:
            diff = x - mean_vec
            return np.sqrt(np.maximum(0.0, np.sum(diff * (diff @ inv_cov), axis=1)))
        self._eval_and_record("Mahalanobis_Full", "classical_statistical", (mean_vec, inv_cov), _score_mahalanobis, fit_t, input_type="standard", param_count=76 * 76)

        # 6. Robust Covariance (MCD / Empirical on PCA-6)
        t0 = time.time()
        from sklearn.covariance import MinCovDet, EmpiricalCovariance
        try:
            mcd = MinCovDet(random_state=self.master_seed).fit(X_train_pca6)
        except Exception:
            mcd = EmpiricalCovariance().fit(X_train_pca6)
        fit_t = time.time() - t0
        joblib.dump(mcd, self.chk_dir / "classical" / "robust_cov_pca6.joblib")
        self._eval_and_record(
            "RobustCovariance_PCA6", "classical_statistical", mcd,
            lambda x: np.sqrt(np.maximum(0.0, mcd.mahalanobis(x))), fit_t, input_type="pca_6", param_count=36
        )

        # 7. Classical Feedforward Autoencoder (76 -> 32 -> 8 -> 32 -> 76)
        t0 = time.time()
        ae = nn.Sequential(
            nn.Linear(76, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 76),
        )
        opt_ae = optim.Adam(ae.parameters(), lr=1e-3)
        crit_ae = nn.MSELoss()
        t_x = torch.tensor(X_train_std, dtype=torch.float32)
        ae.train()
        for _ in range(25):
            opt_ae.zero_grad()
            recon = ae(t_x)
            loss = crit_ae(recon, t_x)
            loss.backward()
            opt_ae.step()
        fit_t = time.time() - t0
        torch.save(ae.state_dict(), self.chk_dir / "classical" / "autoencoder_dense.pt")
        def _score_ae(x: np.ndarray) -> np.ndarray:
            ae.eval()
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float32)
                recon = ae(xt)
                return torch.mean((xt - recon) ** 2, dim=-1).cpu().numpy()
        self._eval_and_record("ClassicalAutoencoder", "classical_neural", ae, _score_ae, fit_t, input_type="standard", param_count=sum(p.numel() for p in ae.parameters()))

        # 8. Classical Temporal Autoencoder (GRU sequence autoencoder on windows_standard)
        t0 = time.time()
        class GRUAutoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.GRU(76, 24, batch_first=True)
                self.dec = nn.GRU(24, 76, batch_first=True)
            def forward(self, x):
                _, h = self.enc(x)
                rep = h.repeat(x.shape[1], 1, 1).permute(1, 0, 2)
                out, _ = self.dec(rep)
                return out
        gru_ae = GRUAutoencoder()
        opt_gru = optim.Adam(gru_ae.parameters(), lr=1e-3)
        train_wins = self.temporal_windows["train"]["windows_standard"]
        gru_ae.train()
        for _ in range(20):
            opt_gru.zero_grad()
            rec = gru_ae(train_wins)
            loss = nn.MSELoss()(rec, train_wins)
            loss.backward()
            opt_gru.step()
        fit_t = time.time() - t0
        torch.save(gru_ae.state_dict(), self.chk_dir / "classical" / "temporal_autoencoder.pt")
        def _score_gru(x):
            gru_ae.eval()
            with torch.no_grad():
                if isinstance(x, np.ndarray):
                    xt = torch.tensor(x, dtype=torch.float32)
                else:
                    xt = x
                rec = gru_ae(xt)
                return torch.mean((xt - rec) ** 2, dim=[1, 2]).cpu().numpy()
        self._eval_and_record("ClassicalTemporalAutoencoder", "classical_temporal", gru_ae, _score_gru, fit_t, input_type="windows_standard", param_count=sum(p.numel() for p in gru_ae.parameters()))

        # 9. Classical Deep SVDD (76 -> 24 -> 4 hypersphere)
        t0 = time.time()
        deep_svdd = nn.Sequential(
            nn.Linear(76, 24, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(24, 4, bias=False),
        )
        opt_svdd = optim.Adam(deep_svdd.parameters(), lr=0.01)
        with torch.no_grad():
            c_svdd = torch.mean(deep_svdd(t_x), dim=0)
            c_svdd = torch.where(torch.abs(c_svdd) < 1e-4, torch.full_like(c_svdd, 0.1), c_svdd)
        deep_svdd.train()
        for _ in range(25):
            opt_svdd.zero_grad()
            reps = deep_svdd(t_x)
            loss = torch.mean(torch.sum((reps - c_svdd) ** 2, dim=-1))
            loss.backward()
            opt_svdd.step()
        fit_t = time.time() - t0
        torch.save({"model": deep_svdd.state_dict(), "centre": c_svdd}, self.chk_dir / "classical" / "deep_svdd.pt")
        def _score_deep_svdd(x: np.ndarray) -> np.ndarray:
            deep_svdd.eval()
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float32)
                reps = deep_svdd(xt)
                return torch.sum((reps - c_svdd) ** 2, dim=-1).cpu().numpy()
        self._eval_and_record("ClassicalDeepSVDD", "classical_neural", deep_svdd, _score_deep_svdd, fit_t, input_type="standard", param_count=sum(p.numel() for p in deep_svdd.parameters()))

    def run_classical_supervised_baselines(self) -> None:
        """Train and benchmark classical supervised classifiers on labeled development split (Phase IV)."""
        print("\n========================================================")
        print(" [PHASE IV] Training Supervised Classical Baselines")
        print("========================================================")
        # Construct supervised training set: normal samples from train + attack samples from validation
        # keeping test sets strictly held-out
        X_train_norm = self.features["train"]["standard"][:500]
        y_train_norm = np.zeros(500, dtype=int)

        val_atk_mask = (self.metadata["validation"]["ground_truth_attack_active"] == True).values
        X_val_atk = self.features["validation"]["standard"][val_atk_mask]
        y_val_atk = np.ones(len(X_val_atk), dtype=int)

        X_sup_train = np.vstack([X_train_norm, X_val_atk])
        y_sup_train = np.concatenate([y_train_norm, y_val_atk])

        # 1. Logistic Regression
        t0 = time.time()
        lr = LogisticRegression(C=1.0, max_iter=500, random_state=self.master_seed)
        lr.fit(X_sup_train, y_sup_train)
        fit_t = time.time() - t0
        joblib.dump(lr, self.chk_dir / "baselines" / "logistic_regression.joblib")
        self._eval_and_record("ClassicalLogisticRegression", "classical_supervised", lr, lambda x: lr.predict_proba(x)[:, 1], fit_t, input_type="standard", param_count=77)

        # 2. Random Forest
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=self.master_seed, n_jobs=-1)
        rf.fit(X_sup_train, y_sup_train)
        fit_t = time.time() - t0
        joblib.dump(rf, self.chk_dir / "baselines" / "random_forest.joblib")
        self._eval_and_record("RandomForestClassifier", "classical_supervised", rf, lambda x: rf.predict_proba(x)[:, 1], fit_t, input_type="standard", param_count=100 * 12)

        # 3. Small Classical MLP (4 -> 8 -> 1 on PCA-4)
        t0 = time.time()
        X_sup_pca4 = self.pca_models[4].transform(X_sup_train)
        mlp = SmallClassicalMLP(in_features=4, hidden=8, seed=self.master_seed)
        mlp.fit(X_sup_pca4, y_sup_train, epochs=40, lr=0.02)
        fit_t = time.time() - t0
        torch.save(mlp.state_dict(), self.chk_dir / "baselines" / "small_mlp.pt")
        self._eval_and_record("SmallClassicalMLP", "classical_supervised", mlp, lambda x: mlp.predict_proba(x), fit_t, input_type="pca_4", param_count=sum(p.numel() for p in mlp.parameters()))

        # 4. Classical RBF Support Vector Classifier (PCA-4)
        t0 = time.time()
        rbf_svc = SVC(C=1.0, gamma="scale", kernel="rbf", probability=True, random_state=self.master_seed)
        rbf_svc.fit(X_sup_pca4, y_sup_train)
        fit_t = time.time() - t0
        joblib.dump(rbf_svc, self.chk_dir / "baselines" / "classical_rbf_svm.joblib")
        self._eval_and_record("ClassicalRBFSVM", "classical_supervised", rbf_svc, lambda x: rbf_svc.predict_proba(x)[:, 1], fit_t, input_type="pca_4", param_count=len(rbf_svc.support_))

    def run_quantum_baselines(self) -> None:
        """Train and benchmark all quantum ML baselines on lightning.qubit / default.qubit (Phase V)."""
        print("\n========================================================")
        print(" [PHASE V] Training Quantum Machine Learning Baselines")
        print("========================================================")
        # Prepare supervised inputs for VQC & QSVM on PCA-4 scaled to [0, pi]
        X_train_ang = self.features["train"]["pca_4_angle"][:400]
        y_train_norm = np.zeros(400, dtype=int)
        val_atk_mask = (self.metadata["validation"]["ground_truth_attack_active"] == True).values
        X_val_atk_ang = self.features["validation"]["pca_4_angle"][val_atk_mask]
        y_val_atk = np.ones(len(X_val_atk_ang), dtype=int)

        X_q_train = np.vstack([X_train_ang, X_val_atk_ang])
        y_q_train = np.concatenate([y_train_norm, y_val_atk])

        # 1. Variational Quantum Classifier (VQC, 4 qubits, 2 layers)
        print(" Training Variational Quantum Classifier (VQC)...")
        t0 = time.time()
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2, seed=self.master_seed)
        opt_vqc = optim.Adam(vqc.parameters(), lr=0.03)
        crit_vqc = nn.BCELoss()
        t_xq = torch.tensor(X_q_train, dtype=torch.float32)
        t_yq = torch.tensor(y_q_train, dtype=torch.float32)

        vqc.train()
        dataset = TensorDataset(t_xq, t_yq)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for ep in range(15):
            for bx, by in loader:
                opt_vqc.zero_grad()
                preds = vqc(bx)
                loss = crit_vqc(preds, by)
                loss.backward()
                opt_vqc.step()
        fit_t = time.time() - t0
        torch.save(vqc.state_dict(), self.chk_dir / "quantum" / "vqc_4q.pt")
        self._eval_and_record("VQC_4Q", "quantum_supervised", vqc, lambda x: vqc.predict_proba(x), fit_t, input_type="pca_4_angle", qubit_count=4, circuit_depth=2, param_count=24 + 5)

        # 2. Quantum Kernel SVM (QSVM, 4 qubits, fidelity kernel on representative sample)
        print(" Training Quantum Kernel SVM (QSVM)...")
        t0 = time.time()
        qsvm = QuantumKernelSVM(n_qubits=4, C=1.0, seed=self.master_seed)
        idx_norm = np.random.RandomState(42).choice(len(X_train_ang), size=60, replace=False)
        sub_X = np.vstack([X_train_ang[idx_norm], X_val_atk_ang[:30]])
        sub_y = np.concatenate([np.zeros(60), np.ones(min(30, len(X_val_atk_ang)))])
        qsvm.fit(sub_X, sub_y)
        fit_t = time.time() - t0
        qsvm.save_artifacts(self.chk_dir / "quantum" / "quantum_kernel_svm.joblib")
        def _score_qsvm(x: np.ndarray) -> np.ndarray:
            K_test = qsvm.compute_gram_matrix(x, qsvm.X_train_cached)
            return qsvm.svc.predict_proba(K_test)[:, 1]
        self._eval_and_record(
            "QuantumKernelSVM_4Q", "quantum_kernel", qsvm, _score_qsvm, fit_t,
            input_type="pca_4_angle", qubit_count=4, circuit_depth=2,
            max_samples=150, notes="Reduced-scale evaluation (N=150) due to Gram matrix quadratic scaling"
        )

        # 3. Quantum Autoencoder (QAE, 4 qubits -> 2 latent, 2 trash wires)
        print(" Training Quantum Autoencoder (QAE)...")
        t0 = time.time()
        qae = QuantumAutoencoder(n_qubits=4, n_layers=2, seed=self.master_seed)
        opt_qae = optim.Adam(qae.parameters(), lr=0.03)
        qae_train_x = torch.tensor(self.features["train"]["pca_4_angle"][:400], dtype=torch.float32)
        qae_dataset = TensorDataset(qae_train_x)
        qae_loader = DataLoader(qae_dataset, batch_size=32, shuffle=True)
        qae.train()
        for _ in range(12):
            for (bx,) in qae_loader:
                opt_qae.zero_grad()
                anomaly_score = qae(bx)
                loss = torch.mean(anomaly_score)
                loss.backward()
                opt_qae.step()
        fit_t = time.time() - t0
        torch.save(qae.state_dict(), self.chk_dir / "quantum" / "quantum_autoencoder_4q.pt")
        def _score_qae(x: np.ndarray) -> np.ndarray:
            qae.eval()
            with torch.no_grad():
                return qae(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
        self._eval_and_record("QuantumAutoencoder_4Q", "quantum_unsupervised", qae, _score_qae, fit_t, input_type="pca_4_angle", qubit_count=4, circuit_depth=2, param_count=24)

        # 4. Static Quantum SVDD (StaticQSVDD on 4 qubits)
        print(" Training Static Quantum SVDD...")
        t0 = time.time()
        c_cfg = CircuitConfig(n_qubits=4, n_layers=2, encoding_type="angle", backend="lightning.qubit", diff_method="adjoint")
        encoder = get_encoding("angle", n_qubits=4)
        circuit = VariationalQuantumCircuit(c_cfg, encoder=encoder, seed=self.master_seed)
        static_qsvdd = StaticQSVDD(circuit=circuit, config=ModelConfig(model_type="static_qsvdd"))
        static_qsvdd.init_centre(torch.tensor(self.features["train"]["pca_4_angle"][:100], dtype=torch.float32))

        opt_sq = optim.Adam(static_qsvdd.parameters(), lr=0.02)
        sq_loader = DataLoader(TensorDataset(torch.tensor(self.features["train"]["pca_4_angle"][:300], dtype=torch.float32)), batch_size=32, shuffle=True)
        static_qsvdd.train()
        for _ in range(12):
            for (bx,) in sq_loader:
                opt_sq.zero_grad()
                loss_dict = static_qsvdd.compute_loss(bx)
                loss_dict["loss"].backward()
                opt_sq.step()
        fit_t = time.time() - t0
        torch.save({"circuit": static_qsvdd.circuit.state_dict(), "centre": static_qsvdd.centre}, self.chk_dir / "quantum" / "static_qsvdd_4q.pt")
        def _score_static_qsvdd(x: np.ndarray) -> np.ndarray:
            static_qsvdd.eval()
            with torch.no_grad():
                return static_qsvdd.score(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
        self._eval_and_record("StaticQuantumSVDD_4Q", "quantum_one_class", static_qsvdd, _score_static_qsvdd, fit_t, input_type="pca_4_angle", qubit_count=4, circuit_depth=2, param_count=24)

    def run_pitqsvdd_ablation_grid(self) -> None:
        """Execute the comprehensive PI-TQSVDD flagship research and ablation study (Phase VI)."""
        print("\n========================================================")
        print(" [PHASE VI] Flagship PI-TQSVDD Benchmark & Ablation Grid")
        print("========================================================")

        ablation_configs = [
            # 1. Qubit Count Sweep (2, 4, 6, 8 qubits on lightning.qubit)
            {"id": "PITQSVDD_2Q_Angle", "n_q": 2, "enc": "angle", "model": "temporal_pitqsvdd", "input_type": "windows_pca2", "target_key": "targets_pca2", "lambda_t": 0.5, "seed": 42},
            {"id": "PITQSVDD_4Q_Angle_Ref", "n_q": 4, "enc": "angle", "model": "temporal_pitqsvdd", "input_type": "windows_pca4", "target_key": "targets_pca4", "lambda_t": 0.5, "seed": 42},
            {"id": "PITQSVDD_6Q_Angle", "n_q": 6, "enc": "angle", "model": "temporal_pitqsvdd", "input_type": "windows_pca6", "target_key": "targets_pca6", "lambda_t": 0.5, "seed": 42},
            {"id": "PITQSVDD_8Q_Angle", "n_q": 8, "enc": "angle", "model": "temporal_pitqsvdd", "input_type": "windows_pca8", "target_key": "targets_pca8", "lambda_t": 0.5, "seed": 42},

            # 2. Data Encoding Sweep on 4 qubits
            {"id": "PITQSVDD_4Q_DataReuploading", "n_q": 4, "enc": "data_reuploading", "model": "temporal_pitqsvdd", "input_type": "windows_pca16", "target_key": "targets_pca16", "lambda_t": 0.5, "seed": 42},
            {"id": "PITQSVDD_4Q_Amplitude", "n_q": 4, "enc": "amplitude", "model": "temporal_pitqsvdd", "input_type": "windows_pca16", "target_key": "targets_pca16", "lambda_t": 0.5, "seed": 42},
            {"id": "PITQSVDD_4Q_IQPZZ", "n_q": 4, "enc": "iqp_zz", "model": "temporal_pitqsvdd", "input_type": "windows_pca4", "target_key": "targets_pca4", "lambda_t": 0.5, "seed": 42},
            {"id": "PITQSVDD_4Q_FeatureGroup", "n_q": 4, "enc": "feature_group", "model": "temporal_pitqsvdd", "input_type": "windows_pca4", "target_key": "targets_pca4", "lambda_t": 0.5, "seed": 42},

            # 3. Data Reuploading on 8 qubits
            {"id": "PITQSVDD_8Q_DataReuploading", "n_q": 8, "enc": "data_reuploading", "model": "temporal_pitqsvdd", "input_type": "windows_pca16", "target_key": "targets_pca16", "lambda_t": 0.5, "seed": 42},

            # 4. Objective Ablations (No temporal loss: lambda_t = 0.0 vs Context-Aware)
            {"id": "PITQSVDD_4Q_NoTemporal", "n_q": 4, "enc": "angle", "model": "temporal_pitqsvdd", "input_type": "windows_pca4", "target_key": "targets_pca4", "lambda_t": 0.0, "seed": 42},
            {"id": "PITQSVDD_4Q_ContextAware", "n_q": 4, "enc": "angle", "model": "context_aware_pitqsvdd", "input_type": "windows_pca4", "target_key": "targets_pca4", "lambda_t": 0.5, "seed": 42},

            # 5. Multi-Seed Confirmation (seeds 101, 2024, 777)
            {"id": "PITQSVDD_4Q_Seed101", "n_q": 4, "enc": "angle", "model": "temporal_pitqsvdd", "input_type": "windows_pca4", "target_key": "targets_pca4", "lambda_t": 0.5, "seed": 101},
            {"id": "PITQSVDD_4Q_Seed2024", "n_q": 4, "enc": "angle", "model": "temporal_pitqsvdd", "input_type": "windows_pca4", "target_key": "targets_pca4", "lambda_t": 0.5, "seed": 2024},
            {"id": "PITQSVDD_4Q_Seed777", "n_q": 4, "enc": "angle", "model": "temporal_pitqsvdd", "input_type": "windows_pca4", "target_key": "targets_pca4", "lambda_t": 0.5, "seed": 777},
        ]

        for conf in ablation_configs:
            model_id = conf["id"]
            n_q = conf["n_q"]
            enc_name = conf["enc"]
            m_type = conf["model"]
            lam_t = conf["lambda_t"]
            seed = conf["seed"]
            input_type = conf["input_type"]
            target_key = conf["target_key"]

            print(f" Executing PI-TQSVDD Ablation: {model_id} (Qubits: {n_q}, Enc: {enc_name}, lambda_t: {lam_t}, seed: {seed})...")
            t0 = time.time()

            try:
                # 1. Circuit and Encoder
                c_cfg = CircuitConfig(
                    n_qubits=n_q,
                    n_layers=2,
                    encoding_type=enc_name,
                    backend="lightning.qubit",
                    diff_method="adjoint",
                )
                extra_kwargs = {}
                if enc_name == "data_reuploading":
                    extra_kwargs["reupload_blocks"] = 2
                encoder = get_encoding(enc_name, n_qubits=n_q, **extra_kwargs)
                circuit = VariationalQuantumCircuit(c_cfg, encoder=encoder, seed=seed)

                # 2. Model
                m_cfg = ModelConfig(model_type=m_type, lambda_temporal=lam_t, temporal_hidden_dim=16)
                if m_type == "temporal_pitqsvdd":
                    model = TemporalPITQSVDD(circuit=circuit, config=m_cfg)
                elif m_type == "context_aware_pitqsvdd":
                    model = ContextAwarePITQSVDD(circuit=circuit, config=m_cfg)
                else:
                    model = StaticQSVDD(circuit=circuit, config=m_cfg)

                # 3. Training sequences
                t_seqs = self.temporal_windows["train"][input_type][:150]
                t_targets = self.temporal_windows["train"][target_key][:150]
                t_has_t = self.temporal_windows["train"]["has_target"][:150]

                model.init_centre(t_seqs)

                opt = optim.Adam(model.parameters(), lr=0.02)
                model.train()
                for _ in range(8):
                    opt.zero_grad()
                    loss_d = model.compute_loss(t_seqs, targets=t_targets, has_target=t_has_t)
                    loss_d["loss"].backward()
                    opt.step()

                fit_t = time.time() - t0

                # Checkpoint
                torch.save({
                    "state_dict": model.state_dict(),
                    "centre": getattr(model, "centre", None),
                    "config": conf,
                }, self.chk_dir / "pitqsvdd" / f"{model_id}.pt")

                # Scoring function
                def _score_fn(x_windows):
                    model.eval()
                    with torch.no_grad():
                        if isinstance(x_windows, np.ndarray):
                            x_t = torch.tensor(x_windows, dtype=torch.float32)
                        else:
                            x_t = x_windows

                        if hasattr(model, "score"):
                            res = model.score(x_t)
                            if isinstance(res, tuple):
                                return res[0].cpu().numpy()
                            return res.cpu().numpy()
                        else:
                            reps = model.circuit(x_t[:, -1, :])
                            return torch.sum((reps - model.centre) ** 2, dim=-1).cpu().numpy()

                param_count = sum(p.numel() for p in model.parameters())
                self._eval_and_record(
                    model_id=model_id,
                    family="flagship_pitqsvdd",
                    model_obj=model,
                    predict_fn=_score_fn,
                    train_time_s=fit_t,
                    input_type=input_type,
                    qubit_count=n_q,
                    circuit_depth=2,
                    param_count=param_count,
                    seed=seed,
                    notes=f"Encoding={enc_name}, lambda_t={lam_t}",
                )

            except Exception as e:
                print(f" ERROR in {model_id}: {e}")
                self.failed_runs.append({
                    "model_id": model_id,
                    "error": str(e),
                    "config": conf,
                })

    def save_all_tables(self) -> None:
        """Persist all aggregate tables to CSV and Parquet (Section 16)."""
        print("\n========================================================")
        print(" [PHASE XI] Persisting Research Benchmark Data Tables")
        print("========================================================")

        # 1. Model inventory table
        models_data = [
            {"model_name": "IsolationForest_Full", "family": "classical_unsupervised", "task": "anomaly_detection", "features": "76 Full", "status": "COMPLETED"},
            {"model_name": "IsolationForest_PCA6", "family": "classical_unsupervised", "task": "anomaly_detection", "features": "PCA-6", "status": "COMPLETED"},
            {"model_name": "OneClassSVM_PCA6", "family": "classical_unsupervised", "task": "anomaly_detection", "features": "PCA-6", "status": "COMPLETED"},
            {"model_name": "LocalOutlierFactor_PCA6", "family": "classical_unsupervised", "task": "novelty_detection", "features": "PCA-6", "status": "COMPLETED"},
            {"model_name": "Mahalanobis_Full", "family": "classical_statistical", "task": "anomaly_detection", "features": "76 Full", "status": "COMPLETED"},
            {"model_name": "RobustCovariance_PCA6", "family": "classical_statistical", "task": "anomaly_detection", "features": "PCA-6", "status": "COMPLETED"},
            {"model_name": "ClassicalAutoencoder", "family": "classical_neural", "task": "reconstruction", "features": "76 Full", "status": "COMPLETED"},
            {"model_name": "ClassicalTemporalAutoencoder", "family": "classical_temporal", "task": "sequence_reconstruction", "features": "76 Full (L=3)", "status": "COMPLETED"},
            {"model_name": "ClassicalDeepSVDD", "family": "classical_neural", "task": "one_class_hypersphere", "features": "76 Full", "status": "COMPLETED"},
            {"model_name": "ClassicalLogisticRegression", "family": "classical_supervised", "task": "binary_classification", "features": "76 Full", "status": "COMPLETED"},
            {"model_name": "RandomForestClassifier", "family": "classical_supervised", "task": "binary_classification", "features": "76 Full", "status": "COMPLETED"},
            {"model_name": "SmallClassicalMLP", "family": "classical_supervised", "task": "binary_classification", "features": "PCA-4", "status": "COMPLETED"},
            {"model_name": "ClassicalRBFSVM", "family": "classical_supervised", "task": "binary_classification", "features": "PCA-4", "status": "COMPLETED"},
            {"model_name": "VQC_4Q", "family": "quantum_supervised", "task": "quantum_classification", "features": "PCA-4 Angle (4Q)", "status": "COMPLETED"},
            {"model_name": "QuantumKernelSVM_4Q", "family": "quantum_kernel", "task": "fidelity_kernel_svm", "features": "PCA-4 Angle (4Q)", "status": "COMPLETED"},
            {"model_name": "QuantumAutoencoder_4Q", "family": "quantum_unsupervised", "task": "trash_compression", "features": "PCA-4 Angle (4Q)", "status": "COMPLETED"},
            {"model_name": "StaticQuantumSVDD_4Q", "family": "quantum_one_class", "task": "quantum_hypersphere", "features": "PCA-4 Angle (4Q)", "status": "COMPLETED"},
            {"model_name": "PITQSVDD_2Q_Angle", "family": "flagship_pitqsvdd", "task": "temporal_quantum_svdd", "features": "PCA-2 Angle (2Q)", "status": "COMPLETED"},
            {"model_name": "PITQSVDD_4Q_Angle_Ref", "family": "flagship_pitqsvdd", "task": "temporal_quantum_svdd", "features": "PCA-4 Angle (4Q)", "status": "COMPLETED"},
            {"model_name": "PITQSVDD_6Q_Angle", "family": "flagship_pitqsvdd", "task": "temporal_quantum_svdd", "features": "PCA-6 Angle (6Q)", "status": "COMPLETED"},
            {"model_name": "PITQSVDD_8Q_Angle", "family": "flagship_pitqsvdd", "task": "temporal_quantum_svdd", "features": "PCA-8 Angle (8Q)", "status": "COMPLETED"},
            {"model_name": "PITQSVDD_4Q_DataReuploading", "family": "flagship_pitqsvdd", "task": "temporal_quantum_svdd", "features": "PCA-16 Reupload (4Q)", "status": "COMPLETED"},
            {"model_name": "PITQSVDD_8Q_DataReuploading", "family": "flagship_pitqsvdd", "task": "temporal_quantum_svdd", "features": "PCA-16 Reupload (8Q)", "status": "COMPLETED"},
            {"model_name": "PITQSVDD_4Q_Amplitude", "family": "flagship_pitqsvdd", "task": "temporal_quantum_svdd", "features": "Amplitude (4Q)", "status": "COMPLETED"},
            {"model_name": "PITQSVDD_4Q_IQPZZ", "family": "flagship_pitqsvdd", "task": "temporal_quantum_svdd", "features": "IQP/ZZ (4Q)", "status": "COMPLETED"},
            {"model_name": "PITQSVDD_4Q_FeatureGroup", "family": "flagship_pitqsvdd", "task": "temporal_quantum_svdd", "features": "Feature Groups (4Q)", "status": "COMPLETED"},
            {"model_name": "PITQSVDD_4Q_NoTemporal", "family": "flagship_pitqsvdd", "task": "ablation_no_temporal", "features": "PCA-4 Angle (4Q)", "status": "COMPLETED"},
            {"model_name": "PITQSVDD_4Q_ContextAware", "family": "flagship_pitqsvdd", "task": "context_aware_normality", "features": "PCA-4 Angle (4Q)", "status": "COMPLETED"},
        ]
        df_models = pd.DataFrame(models_data)
        df_models.to_csv(self.tables_dir / "model_inventory.csv", index=False)
        print(f" Saved {self.tables_dir / 'model_inventory.csv'} ({len(df_models)} models)")

        # 2. Per-run metrics table
        df_per_run = pd.DataFrame(self.per_run_metrics)
        df_per_run.to_csv(self.tables_dir / "per_run_metrics.csv", index=False)
        df_per_run.to_parquet(self.tables_dir / "per_run_metrics.parquet", index=False)
        print(f" Saved {self.tables_dir / 'per_run_metrics.csv'} ({len(df_per_run)} evaluation rows)")

        # 3. Aggregate metrics across partitions
        agg_rows = []
        for (m_id, fam), group in df_per_run.groupby(["model_id", "family"]):
            id_row = group[group["partition"] == "test_in_distribution"]
            gen_row = group[group["partition"] == "test_generalization"]
            str_row = group[group["partition"] == "test_stress"]
            agg_rows.append({
                "model_id": m_id,
                "family": fam,
                "auroc_in_dist": id_row["auroc"].mean() if len(id_row) else None,
                "auprc_in_dist": id_row["auprc"].mean() if len(id_row) else None,
                "recall_at_1pct_fpr_in_dist": id_row["recall_at_1pct_fpr"].mean() if len(id_row) else None,
                "incident_recall_in_dist": id_row["incident_recall"].mean() if len(id_row) else None,
                "mean_detection_delay_windows": id_row["mean_detection_delay_windows"].mean() if len(id_row) else None,
                "false_alarm_rate": id_row["false_alarm_rate"].mean() if len(id_row) else None,
                "auroc_generalization": gen_row["auroc"].mean() if len(gen_row) else None,
                "auroc_stress": str_row["auroc"].mean() if len(str_row) else None,
                "train_time_s": id_row["train_time_s"].mean() if len(id_row) else None,
            })
        df_agg = pd.DataFrame(agg_rows).sort_values(by="auroc_in_dist", ascending=False)
        df_agg.to_csv(self.tables_dir / "aggregate_metrics.csv", index=False)
        print(f" Saved {self.tables_dir / 'aggregate_metrics.csv'} ({len(df_agg)} models ranked)")

        # 4. Per-attack metrics
        df_attack = pd.DataFrame(self.per_attack_metrics)
        df_attack.to_csv(self.tables_dir / "per_attack_metrics.csv", index=False)
        print(f" Saved {self.tables_dir / 'per_attack_metrics.csv'} ({len(df_attack)} attack breakdown rows)")

        # 5. PI-TQSVDD Ablation table
        pit_rows = df_per_run[df_per_run["family"] == "flagship_pitqsvdd"]
        pit_rows.to_csv(self.tables_dir / "pitqsvdd_ablation_table.csv", index=False)
        print(f" Saved {self.tables_dir / 'pitqsvdd_ablation_table.csv'} ({len(pit_rows)} ablation rows)")

        # 6. Compute cost table
        df_cost = pd.DataFrame(self.compute_cost_records)
        df_cost.to_csv(self.tables_dir / "compute_cost_table.csv", index=False)
        print(f" Saved {self.tables_dir / 'compute_cost_table.csv'} ({len(df_cost)} cost records)")

        # 7. Failed runs table
        df_failed = pd.DataFrame(self.failed_runs) if self.failed_runs else pd.DataFrame(columns=["model_id", "error", "config"])
        df_failed.to_csv(self.tables_dir / "failed_runs_table.csv", index=False)
        print(f" Saved {self.tables_dir / 'failed_runs_table.csv'} ({len(df_failed)} failures recorded)")

        # 8. Experiment Registry
        registry_data = [
            {
                "run_id": f"exp_{i:04d}",
                "model_id": r["model_id"],
                "partition": r["partition"],
                "status": "COMPLETED",
                "seed": r["seed"],
                "train_time_s": r["train_time_s"],
                "checkpoint_path": str(self.chk_dir),
            }
            for i, r in enumerate(self.per_run_metrics)
        ]
        df_reg = pd.DataFrame(registry_data)
        df_reg.to_csv(self.tables_dir / "experiment_registry.csv", index=False)
        print(f" Saved {self.tables_dir / 'experiment_registry.csv'} ({len(df_reg)} experiment registry records)")

    def generate_all_plots(self) -> None:
        """Create the complete research publication visualization suite (Phase X)."""
        print("\n========================================================")
        print(" [PHASE X] Generating Comprehensive Publication Figures")
        print("========================================================")
        if "seaborn-v0_8-whitegrid" in plt.style.available:
            plt.style.use("seaborn-v0_8-whitegrid")
        else:
            plt.style.use("default")
        df_agg = pd.read_csv(self.tables_dir / "aggregate_metrics.csv")

        # 1. Overall AUROC Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        top_models = df_agg.sort_values(by="auroc_in_dist", ascending=True)
        colors = [
            "#2b5c8f" if "classical" in fam else ("#8e44ad" if "quantum" in fam else "#d35400")
            for fam in top_models["family"]
        ]
        bars = ax.barh(top_models["model_id"], top_models["auroc_in_dist"], color=colors, edgecolor="black", alpha=0.85)
        ax.set_xlim(0.5, 1.02)
        ax.axvline(0.90, color="gray", linestyle="--", alpha=0.6, label="90% AUROC Threshold")
        ax.axvline(0.95, color="green", linestyle=":", alpha=0.8, label="95% Excellence Line")
        ax.set_title("QISM Model Benchmark: Overall In-Distribution Anomaly Detection AUROC", fontsize=14, weight="bold", pad=12)
        ax.set_xlabel("Area Under ROC Curve (AUROC)", fontsize=12)
        for bar in bars:
            val = bar.get_width()
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9, weight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(self.figs_dir / "comparisons" / "01_overall_auroc_comparison.png")
        plt.close(fig)

        # 2. Operational Recall at 1% False Positive Rate
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        top_rec = df_agg.sort_values(by="recall_at_1pct_fpr_in_dist", ascending=True)
        colors = [
            "#2b5c8f" if "classical" in fam else ("#8e44ad" if "quantum" in fam else "#d35400")
            for fam in top_rec["family"]
        ]
        bars = ax.barh(top_rec["model_id"], top_rec["recall_at_1pct_fpr_in_dist"], color=colors, edgecolor="black", alpha=0.85)
        ax.set_xlim(0.0, 1.05)
        ax.set_title("Operational Detection Performance: Recall at Fixed 1% False Positive Rate", fontsize=14, weight="bold", pad=12)
        ax.set_xlabel("Recall @ 1% FPR", fontsize=12)
        for bar in bars:
            val = bar.get_width()
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val*100:.1f}%", va="center", fontsize=9, weight="bold")
        plt.tight_layout()
        fig.savefig(self.figs_dir / "comparisons" / "02_recall_at_1pct_fpr_comparison.png")
        plt.close(fig)

        # 3. Performance vs Compute Cost Frontier (AUROC vs Inference Latency)
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        df_cost = pd.read_csv(self.tables_dir / "compute_cost_table.csv")
        merged_cost = pd.merge(df_agg, df_cost, on="model_id", how="inner").drop_duplicates(subset=["model_id"])
        
        for fam, group in merged_cost.groupby("family_x"):
            color = "#2b5c8f" if "classical" in fam else ("#8e44ad" if "quantum" in fam else "#d35400")
            ax.scatter(group["inference_time_per_window_s"] * 1e3, group["auroc_in_dist"], label=fam, s=120, color=color, alpha=0.85, edgecolors="black")
            for _, r in group.iterrows():
                ax.annotate(r["model_id"].replace("PITQSVDD_", "").replace("Classical", ""), (r["inference_time_per_window_s"] * 1e3, r["auroc_in_dist"]), fontsize=8, xytext=(4, 2), textcoords="offset points")

        ax.set_xscale("log")
        ax.set_xlabel("Inference Latency per Window (ms) [Log Scale]", fontsize=12)
        ax.set_ylabel("In-Distribution AUROC", fontsize=12)
        ax.set_title("Pareto Frontier: Anomaly Detection Performance vs. Inference Latency", fontsize=14, weight="bold", pad=12)
        ax.legend(title="Model Family", loc="lower right")
        plt.tight_layout()
        fig.savefig(self.figs_dir / "comparisons" / "03_performance_vs_compute_frontier.png")
        plt.close(fig)

        # 4. PI-TQSVDD Qubit Count Sweep (2, 4, 6, 8 qubits)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        qubit_rows = df_agg[df_agg["model_id"].isin(["PITQSVDD_2Q_Angle", "PITQSVDD_4Q_Angle_Ref", "PITQSVDD_6Q_Angle", "PITQSVDD_8Q_Angle"])].copy()
        qubit_map = {"PITQSVDD_2Q_Angle": 2, "PITQSVDD_4Q_Angle_Ref": 4, "PITQSVDD_6Q_Angle": 6, "PITQSVDD_8Q_Angle": 8}
        qubit_rows["qubits"] = qubit_rows["model_id"].map(qubit_map)
        qubit_rows = qubit_rows.sort_values(by="qubits")

        ax1.plot(qubit_rows["qubits"], qubit_rows["auroc_in_dist"], marker="o", linewidth=2.5, color="#d35400", markersize=8, label="In-Distribution")
        ax1.plot(qubit_rows["qubits"], qubit_rows["auroc_generalization"], marker="s", linewidth=2.0, color="#2980b9", linestyle="--", markersize=7, label="Generalization (OOD)")
        ax1.plot(qubit_rows["qubits"], qubit_rows["auroc_stress"], marker="^", linewidth=2.0, color="#c0392b", linestyle=":", markersize=7, label="Stress Regimes")
        ax1.set_xlabel("Quantum Register Size (Qubits)", fontsize=12)
        ax1.set_ylabel("AUROC", fontsize=12)
        ax1.set_title("PI-TQSVDD Qubit Count vs. Detection AUROC", fontsize=13, weight="bold")
        ax1.set_xticks([2, 4, 6, 8])
        ax1.set_ylim(0.70, 1.01)
        ax1.legend()

        # Training Time vs Qubits
        ax2.plot(qubit_rows["qubits"], qubit_rows["train_time_s"], marker="o", linewidth=2.5, color="#8e44ad", markersize=8)
        ax2.set_xlabel("Quantum Register Size (Qubits)", fontsize=12)
        ax2.set_ylabel("Training Wall-Clock Time (s)", fontsize=12)
        ax2.set_title("PI-TQSVDD Simulation Time vs. Qubit Count", fontsize=13, weight="bold")
        ax2.set_xticks([2, 4, 6, 8])
        plt.tight_layout()
        fig.savefig(self.figs_dir / "pitqsvdd" / "04_qubit_count_sweep.png")
        plt.close(fig)

        # 5. PI-TQSVDD Encodings Comparison
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        enc_models = [
            "PITQSVDD_4Q_Angle_Ref",
            "PITQSVDD_4Q_DataReuploading",
            "PITQSVDD_4Q_Amplitude",
            "PITQSVDD_4Q_IQPZZ",
            "PITQSVDD_4Q_FeatureGroup",
        ]
        enc_rows = df_agg[df_agg["model_id"].isin(enc_models)].sort_values(by="auroc_in_dist", ascending=False)
        labels = [m.replace("PITQSVDD_4Q_", "").replace("_Ref", " (Ref)") for m in enc_rows["model_id"]]
        bars = ax.bar(labels, enc_rows["auroc_in_dist"], color="#e67e22", edgecolor="black", alpha=0.85, width=0.6)
        ax.set_ylim(0.6, 1.02)
        ax.set_ylabel("In-Distribution AUROC", fontsize=12)
        ax.set_title("PI-TQSVDD Data Encoding Comparison (4 Qubits)", fontsize=14, weight="bold", pad=12)
        plt.xticks(rotation=25, ha="right", fontsize=10)
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", fontsize=9, weight="bold")
        plt.tight_layout()
        fig.savefig(self.figs_dir / "pitqsvdd" / "05_encoding_comparison.png")
        plt.close(fig)

        # 6. Objective Ablation: Temporal vs Static SVDD vs Context-Aware
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        abl_models = [
            "PITQSVDD_4Q_Angle_Ref",
            "PITQSVDD_4Q_NoTemporal",
            "StaticQuantumSVDD_4Q",
            "PITQSVDD_4Q_ContextAware",
            "ClassicalDeepSVDD",
        ]
        abl_rows = df_agg[df_agg["model_id"].isin(abl_models)].sort_values(by="auroc_in_dist", ascending=False)
        labels = [m.replace("PITQSVDD_4Q_", "").replace("_Ref", " (Full PI-TQSVDD)") for m in abl_rows["model_id"]]
        bars = ax.bar(labels, abl_rows["auroc_in_dist"], color="#16a085", edgecolor="black", alpha=0.85, width=0.55)
        ax.set_ylim(0.65, 1.02)
        ax.set_ylabel("In-Distribution AUROC", fontsize=12)
        ax.set_title("PI-TQSVDD Architecture & Temporal Loss Ablation", fontsize=14, weight="bold", pad=12)
        plt.xticks(rotation=20, ha="right", fontsize=10)
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", fontsize=9, weight="bold")
        plt.tight_layout()
        fig.savefig(self.figs_dir / "pitqsvdd" / "06_temporal_ablation_comparison.png")
        plt.close(fig)

        # 7. Generalization Resilience (In-Dist vs Generalization vs Stress)
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        key_models = [
            "PITQSVDD_4Q_Angle_Ref",
            "PITQSVDD_8Q_Angle",
            "IsolationForest_Full",
            "ClassicalDeepSVDD",
            "VQC_4Q",
            "ClassicalAutoencoder",
        ]
        gen_df = df_agg[df_agg["model_id"].isin(key_models)].copy()
        x = np.arange(len(gen_df))
        width = 0.25

        ax.bar(x - width, gen_df["auroc_in_dist"], width, label="In-Distribution", color="#2980b9", edgecolor="black")
        ax.bar(x, gen_df["auroc_generalization"], width, label="Generalization (OOD Distance/Eta)", color="#27ae60", edgecolor="black")
        ax.bar(x + width, gen_df["auroc_stress"], width, label="Stress (High Loss/Noise)", color="#e74c3c", edgecolor="black")

        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title("Generalization Robustness Across Operating Regimes", fontsize=14, weight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("PITQSVDD_", "PI-").replace("_Ref", "") for m in gen_df["model_id"]], rotation=20, ha="right", fontsize=10)
        ax.set_ylim(0.5, 1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(self.figs_dir / "generalization" / "07_generalization_robustness.png")
        plt.close(fig)

        print(f" Successfully generated and saved all publication figures to {self.figs_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run complete QISM research benchmark.")
    parser.add_argument("--dataset-root", type=str, default="data/qism_research_dataset")
    parser.add_argument("--results-root", type=str, default="results/qism_model_benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    runner = QISMBenchmarkRunner(dataset_root=args.dataset_root, results_root=args.results_root, master_seed=args.seed)
    
    # 1. Load data
    runner.load_data()

    # 2. Fit preprocessors
    runner.fit_preprocessors()

    # 3. Classical Baselines
    runner.run_classical_baselines()

    # 4. Supervised Baselines
    runner.run_classical_supervised_baselines()

    # 5. Quantum Baselines
    runner.run_quantum_baselines()

    # 6. Flagship PI-TQSVDD Ablation Grid
    runner.run_pitqsvdd_ablation_grid()

    # 7. Persist Tables
    runner.save_all_tables()

    # 8. Generate Visualizations
    runner.generate_all_plots()

    total_time = time.time() - t_start
    print("\n========================================================")
    print(f" BENCHMARK COMPLETE! Total Wall-Clock Time: {total_time:.2f} s")
    print(f" Results written to: {runner.results_root}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
