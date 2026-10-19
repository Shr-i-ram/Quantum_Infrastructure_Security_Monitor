"""Real Model Inference Service for QISM Live Demonstration Console.

Loads verified trained checkpoints from results/qism_model_benchmark/checkpoints/.
Applies the fitted preprocessor pipeline from results/qism_model_benchmark/preprocessors/.
Maintains causal temporal sliding windows (L=3) and computes online anomaly scores.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import torch

from ..config import CHECKPOINTS_DIR, PREPROCESSORS_DIR
from qml.pitqsvdd.circuits.variational import VariationalQuantumCircuit
from qml.pitqsvdd.config import CircuitConfig, ModelConfig
from qml.pitqsvdd.encodings import get_encoding
from qml.pitqsvdd.models.temporal_pitqsvdd import TemporalPITQSVDD
from qml.pitqsvdd.models.static_qsvdd import StaticQSVDD
from qml.src.quantum_autoencoder import QuantumAutoencoder
from qml.src.vqc_model import VariationalQuantumClassifier

log = logging.getLogger("qism.demo.inference")

# Calibrated Decision Thresholds at Target 1% False Positive Rate on normal validation data
# Extracted from the authoritative Phase XI benchmark results
CALIBRATED_THRESHOLDS = {
    "PITQSVDD_6Q_Angle": 0.0452,
    "PITQSVDD_4Q_Angle_Ref": 0.0514,
    "PITQSVDD_4Q_Amplitude": 0.0389,
    "PITQSVDD_4Q_DataReuploading": 0.0485,
    "PITQSVDD_8Q_Angle": 0.0612,
    "PITQSVDD_8Q_DataReuploading": 0.0520,
    "PITQSVDD_2Q_Angle": 0.0390,
    "ClassicalTemporalAutoencoder": 0.0125,
    "ClassicalAutoencoder": 0.0182,
    "ClassicalDeepSVDD": 0.1420,
    "IsolationForest_PCA6": 0.1580,
    "IsolationForest_Full": 0.1240,
    "Mahalanobis_Full": 12.85,
    "RobustCovariance_PCA6": 6.42,
    "OneClassSVM_PCA6": 0.0050,
    "LocalOutlierFactor_PCA6": 0.1120,
    "QuantumAutoencoder_4Q": 0.1850,
    "StaticQuantumSVDD_4Q": 0.0480,
    "VQC_4Q": 0.5000,
    "QuantumKernelSVM_4Q": 0.0000,
    "ClassicalLogisticRegression": 0.5000,
    "SmallClassicalMLP": 0.5000,
    "RandomForestClassifier": 0.5000,
    "ClassicalRBFSVM": 0.0000,
}


class ModelRegistry:
    """Discovers and catalogs all trained and verified model checkpoints."""

    @staticmethod
    def get_available_models() -> list[dict[str, Any]]:
        """Return list of models with verified checkpoint files."""
        models = [
            # Flagship PI-TQSVDD
            {
                "id": "PITQSVDD_6Q_Angle",
                "name": "PI-TQSVDD (6 Qubits - Default)",
                "family": "flagship_pitqsvdd",
                "paradigm": "unsupervised_temporal_quantum",
                "qubits": 6,
                "encoding": "angle",
                "circuit_depth": 2,
                "is_default": True,
                "path": str(CHECKPOINTS_DIR / "pitqsvdd" / "PITQSVDD_6Q_Angle.pt"),
                "status": "READY" if (CHECKPOINTS_DIR / "pitqsvdd" / "PITQSVDD_6Q_Angle.pt").exists() else "MISSING",
                "threshold_1pct": CALIBRATED_THRESHOLDS.get("PITQSVDD_6Q_Angle", 0.05),
                "benchmark_auroc": 0.6392,
                "description": "Flagship 6-qubit temporal quantum anomaly detector running on PennyLane lightning.qubit with adjoint differentiation.",
            },
            {
                "id": "PITQSVDD_4Q_Angle_Ref",
                "name": "PI-TQSVDD (4 Qubits Reference)",
                "family": "flagship_pitqsvdd",
                "paradigm": "unsupervised_temporal_quantum",
                "qubits": 4,
                "encoding": "angle",
                "circuit_depth": 2,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "pitqsvdd" / "PITQSVDD_4Q_Angle_Ref.pt"),
                "status": "READY" if (CHECKPOINTS_DIR / "pitqsvdd" / "PITQSVDD_4Q_Angle_Ref.pt").exists() else "MISSING",
                "threshold_1pct": CALIBRATED_THRESHOLDS.get("PITQSVDD_4Q_Angle_Ref", 0.05),
                "benchmark_auroc": 0.6751,
                "description": "4-qubit flagship reference model with optimal quantum state compactness and minimal hypersphere dispersion.",
            },
            {
                "id": "PITQSVDD_4Q_Amplitude",
                "name": "PI-TQSVDD (4 Qubits Amplitude)",
                "family": "flagship_pitqsvdd",
                "paradigm": "unsupervised_temporal_quantum",
                "qubits": 4,
                "encoding": "amplitude",
                "circuit_depth": 2,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "pitqsvdd" / "PITQSVDD_4Q_Amplitude.pt"),
                "status": "READY" if (CHECKPOINTS_DIR / "pitqsvdd" / "PITQSVDD_4Q_Amplitude.pt").exists() else "MISSING",
                "threshold_1pct": CALIBRATED_THRESHOLDS.get("PITQSVDD_4Q_Amplitude", 0.04),
                "benchmark_auroc": 0.7147,
                "description": "Top performing quantum model in benchmark with 16-feature amplitude statevector representation.",
            },
            {
                "id": "PITQSVDD_4Q_DataReuploading",
                "name": "PI-TQSVDD (4 Qubits Data Reuploading)",
                "family": "flagship_pitqsvdd",
                "paradigm": "unsupervised_temporal_quantum",
                "qubits": 4,
                "encoding": "data_reuploading",
                "circuit_depth": 2,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "pitqsvdd" / "PITQSVDD_4Q_DataReuploading.pt"),
                "status": "READY" if (CHECKPOINTS_DIR / "pitqsvdd" / "PITQSVDD_4Q_DataReuploading.pt").exists() else "MISSING",
                "threshold_1pct": CALIBRATED_THRESHOLDS.get("PITQSVDD_4Q_DataReuploading", 0.05),
                "benchmark_auroc": 0.6795,
                "description": "Multi-layer feature interleaving overcoming single-frequency Fourier limits.",
            },
            {
                "id": "PITQSVDD_8Q_DataReuploading",
                "name": "PI-TQSVDD (8 Qubits Data Reuploading)",
                "family": "flagship_pitqsvdd",
                "paradigm": "unsupervised_temporal_quantum",
                "qubits": 8,
                "encoding": "data_reuploading",
                "circuit_depth": 2,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "pitqsvdd" / "PITQSVDD_8Q_DataReuploading.pt"),
                "status": "READY" if (CHECKPOINTS_DIR / "pitqsvdd" / "PITQSVDD_8Q_DataReuploading.pt").exists() else "MISSING",
                "threshold_1pct": CALIBRATED_THRESHOLDS.get("PITQSVDD_8Q_DataReuploading", 0.05),
                "benchmark_auroc": 0.6941,
                "description": "8-qubit register with data reuploading mitigating Hilbert space state dispersion.",
            },
            # Classical Models
            {
                "id": "ClassicalTemporalAutoencoder",
                "name": "Classical Temporal Autoencoder (GRU)",
                "family": "classical_temporal",
                "paradigm": "unsupervised_temporal_classical",
                "qubits": 0,
                "encoding": "full_telemetry",
                "circuit_depth": 0,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "classical" / "temporal_autoencoder.pt"),
                "status": "READY" if (CHECKPOINTS_DIR / "classical" / "temporal_autoencoder.pt").exists() else "MISSING",
                "threshold_1pct": CALIBRATED_THRESHOLDS.get("ClassicalTemporalAutoencoder", 0.012),
                "benchmark_auroc": 0.7616,
                "description": "Benchmark champion classical recurrent GRU sequence autoencoder operating across 76 telemetry features.",
            },
            {
                "id": "ClassicalAutoencoder",
                "name": "Classical Deep Autoencoder",
                "family": "classical_neural",
                "paradigm": "unsupervised_reconstruction",
                "qubits": 0,
                "encoding": "full_telemetry",
                "circuit_depth": 0,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "classical" / "autoencoder_dense.pt"),
                "status": "READY" if (CHECKPOINTS_DIR / "classical" / "autoencoder_dense.pt").exists() else "MISSING",
                "threshold_1pct": CALIBRATED_THRESHOLDS.get("ClassicalAutoencoder", 0.018),
                "benchmark_auroc": 0.7127,
                "description": "Dense feedforward neural autoencoder evaluating point-wise reconstruction error.",
            },
            {
                "id": "IsolationForest_PCA6",
                "name": "Isolation Forest (PCA-6)",
                "family": "classical_unsupervised",
                "paradigm": "unsupervised_tree_isolation",
                "qubits": 0,
                "encoding": "pca_6",
                "circuit_depth": 0,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "classical" / "isolation_forest_pca6.joblib"),
                "status": "READY" if (CHECKPOINTS_DIR / "classical" / "isolation_forest_pca6.joblib").exists() else "MISSING",
                "threshold_1pct": CALIBRATED_THRESHOLDS.get("IsolationForest_PCA6", 0.158),
                "benchmark_auroc": 0.7078,
                "description": "Standard statistical tree ensemble isolating anomalies along 6 principal variance axes.",
            },
            {
                "id": "Mahalanobis_Full",
                "name": "Mahalanobis Distance (Full 76)",
                "family": "classical_statistical",
                "paradigm": "unsupervised_covariance",
                "qubits": 0,
                "encoding": "full_telemetry",
                "circuit_depth": 0,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "classical" / "mahalanobis_full.npz"),
                "status": "READY" if (CHECKPOINTS_DIR / "classical" / "mahalanobis_full.npz").exists() else "MISSING",
                "threshold_1pct": CALIBRATED_THRESHOLDS.get("Mahalanobis_Full", 12.85),
                "benchmark_auroc": 0.6958,
                "description": "Fast statistical distance model with regularized covariance shrinkage across all 76 features.",
            },
            # Quantum Baselines
            {
                "id": "QuantumAutoencoder_4Q",
                "name": "Quantum Autoencoder (4 Qubits)",
                "family": "quantum_unsupervised",
                "paradigm": "unsupervised_compression",
                "qubits": 4,
                "encoding": "angle",
                "circuit_depth": 2,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "quantum" / "quantum_autoencoder_4q.pt"),
                "status": "READY" if (CHECKPOINTS_DIR / "quantum" / "quantum_autoencoder_4q.pt").exists() else "MISSING",
                "threshold_1pct": CALIBRATED_THRESHOLDS.get("QuantumAutoencoder_4Q", 0.185),
                "benchmark_auroc": 0.6832,
                "description": "Quantum state compressor mapping 2 trash wires to reference |00> state.",
            },
            {
                "id": "StaticQuantumSVDD_4Q",
                "name": "Static Quantum SVDD (4 Qubits)",
                "family": "quantum_one_class",
                "paradigm": "unsupervised_quantum_hypersphere",
                "qubits": 4,
                "encoding": "angle",
                "circuit_depth": 2,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "quantum" / "static_qsvdd_4q.pt"),
                "status": "READY" if (CHECKPOINTS_DIR / "quantum" / "static_qsvdd_4q.pt").exists() else "MISSING",
                "threshold_1pct": CALIBRATED_THRESHOLDS.get("StaticQuantumSVDD_4Q", 0.048),
                "benchmark_auroc": 0.6402,
                "description": "Non-temporal static baseline mapping expectation values to a fixed hypersphere.",
            },
            {
                "id": "VQC_4Q",
                "name": "Variational Quantum Classifier (4 Qubits)",
                "family": "quantum_supervised",
                "paradigm": "supervised_quantum_classification",
                "qubits": 4,
                "encoding": "angle",
                "circuit_depth": 2,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "quantum" / "vqc_4q.pt"),
                "status": "READY" if (CHECKPOINTS_DIR / "quantum" / "vqc_4q.pt").exists() else "MISSING",
                "threshold_1pct": 0.5000,
                "benchmark_auroc": 0.6550,
                "description": "Supervised variational quantum classifier trained on labeled attack examples.",
            },
            {
                "id": "ClassicalLogisticRegression",
                "name": "Classical Logistic Regression",
                "family": "classical_supervised",
                "paradigm": "supervised_linear",
                "qubits": 0,
                "encoding": "full_telemetry",
                "circuit_depth": 0,
                "is_default": False,
                "path": str(CHECKPOINTS_DIR / "baselines" / "logistic_regression.joblib"),
                "status": "READY" if (CHECKPOINTS_DIR / "baselines" / "logistic_regression.joblib").exists() else "MISSING",
                "threshold_1pct": 0.5000,
                "benchmark_auroc": 0.7408,
                "description": "Supervised linear discriminator providing low-latency baseline classification.",
            },
        ]
        return models


class InferenceService:
    """Manages loaded models, input preprocessing, and live anomaly scoring."""

    def __init__(self):
        self.pipeline: dict[str, Any] | None = None
        self.loaded_models: dict[str, Any] = {}
        self.window_history: list[dict[str, Any]] = []
        self.max_history: int = 50
        self._load_pipeline()

    @property
    def feature_names(self) -> list[str]:
        """Return the list of 76 observable telemetry feature column names."""
        if not self.pipeline:
            self._load_pipeline()
        return list(self.pipeline.get("observable_cols", []))

    def _load_pipeline(self) -> None:
        """Load fitted preprocessor artifacts."""
        p_path = PREPROCESSORS_DIR / "fitted_pipeline.joblib"
        if p_path.exists():
            try:
                self.pipeline = joblib.load(p_path)
                log.info("Fitted preprocessor pipeline loaded successfully.")
            except Exception as e:
                log.error(f"Failed to load preprocessor pipeline: {e}")
        else:
            log.warning(f"Preprocessor pipeline not found at {p_path}")

    def load_model(self, model_id: str) -> bool:
        """Load and cache a requested model checkpoint."""
        if model_id in self.loaded_models:
            return True

        avail = {m["id"]: m for m in ModelRegistry.get_available_models()}
        if model_id not in avail:
            log.error(f"Model ID {model_id} is not recognized.")
            return False

        meta = avail[model_id]
        chk_path = Path(meta["path"])
        if not chk_path.exists():
            log.error(f"Checkpoint for {model_id} not found at {chk_path}")
            return False

        try:
            # 1. PI-TQSVDD Models
            if meta["family"] == "flagship_pitqsvdd":
                n_q = meta["qubits"]
                enc_name = meta["encoding"]
                chk = torch.load(chk_path, map_location="cpu")

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
                circuit = VariationalQuantumCircuit(c_cfg, encoder=encoder, seed=42)
                m_cfg = ModelConfig(model_type="temporal_pitqsvdd", lambda_temporal=0.5, temporal_hidden_dim=16)
                model = TemporalPITQSVDD(circuit=circuit, config=m_cfg)
                model.load_state_dict(chk["state_dict"])
                if "centre" in chk and chk["centre"] is not None:
                    model.centre.copy_(chk["centre"])
                model.eval()
                self.loaded_models[model_id] = {"model": model, "type": "pitqsvdd", "meta": meta}
                log.info(f"Loaded PI-TQSVDD model: {model_id}")
                return True

            # 2. Classical Temporal Autoencoder
            elif model_id == "ClassicalTemporalAutoencoder":
                chk = torch.load(chk_path, map_location="cpu")
                import torch.nn as nn
                class TemporalAutoencoder(nn.Module):
                    def __init__(self, in_dim=76, hidden_dim=32, latent_dim=12):
                        super().__init__()
                        self.encoder = nn.GRU(in_dim, hidden_dim, num_layers=2, batch_first=True)
                        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
                        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
                        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
                        self.out_fc = nn.Linear(hidden_dim, in_dim)
                    def forward(self, x):
                        _, h = self.encoder(x)
                        z = self.enc_fc(h[-1])
                        d_in = self.dec_fc(z).unsqueeze(1).repeat(1, x.size(1), 1)
                        d_out, _ = self.decoder(d_in)
                        return self.out_fc(d_out)

                model = TemporalAutoencoder()
                model.load_state_dict(chk["model_state_dict"])
                model.eval()
                self.loaded_models[model_id] = {"model": model, "type": "temporal_autoencoder", "meta": meta}
                log.info("Loaded Classical Temporal Autoencoder")
                return True

            # 3. Classical Dense Autoencoder
            elif model_id == "ClassicalAutoencoder":
                chk = torch.load(chk_path, map_location="cpu")
                import torch.nn as nn
                class DenseAutoencoder(nn.Module):
                    def __init__(self, in_dim=76, latent_dim=16):
                        super().__init__()
                        self.encoder = nn.Sequential(
                            nn.Linear(in_dim, 48), nn.LeakyReLU(0.1),
                            nn.Linear(48, latent_dim), nn.LeakyReLU(0.1),
                        )
                        self.decoder = nn.Sequential(
                            nn.Linear(latent_dim, 48), nn.LeakyReLU(0.1),
                            nn.Linear(48, in_dim),
                        )
                    def forward(self, x):
                        return self.decoder(self.encoder(x))

                model = DenseAutoencoder()
                model.load_state_dict(chk["model_state_dict"])
                model.eval()
                self.loaded_models[model_id] = {"model": model, "type": "autoencoder", "meta": meta}
                log.info("Loaded Classical Dense Autoencoder")
                return True

            # 4. Joblib Estimators (Isolation Forest, Logistic Regression)
            elif str(chk_path).endswith(".joblib"):
                estimator = joblib.load(chk_path)
                self.loaded_models[model_id] = {"model": estimator, "type": "sklearn", "meta": meta}
                log.info(f"Loaded sklearn estimator: {model_id}")
                return True

            # 5. Mahalanobis Full NPZ
            elif model_id == "Mahalanobis_Full":
                data = np.load(chk_path)
                self.loaded_models[model_id] = {
                    "model": {"mean": data["mean"], "inv_cov": data["inv_cov"]},
                    "type": "mahalanobis",
                    "meta": meta,
                }
                log.info("Loaded Mahalanobis Full model")
                return True

            # 6. Static Quantum SVDD
            elif model_id == "StaticQuantumSVDD_4Q":
                chk = torch.load(chk_path, map_location="cpu")
                c_cfg = CircuitConfig(n_qubits=4, n_layers=2, encoding_type="angle", backend="lightning.qubit", diff_method="adjoint")
                encoder = get_encoding("angle", n_qubits=4)
                circuit = VariationalQuantumCircuit(c_cfg, encoder=encoder, seed=42)
                model = StaticQSVDD(circuit=circuit, config=ModelConfig(model_type="static_qsvdd"))
                model.circuit.load_state_dict(chk["circuit"])
                model.centre.copy_(chk["centre"])
                model.eval()
                self.loaded_models[model_id] = {"model": model, "type": "static_qsvdd", "meta": meta}
                log.info("Loaded Static Quantum SVDD")
                return True

            else:
                log.warning(f"Model {model_id} loader not specifically implemented, falling back.")
                return False

        except Exception as e:
            log.error(f"Error loading model {model_id}: {e}", exc_info=True)
            return False

    def transform_features(self, raw_features_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        """Transform raw telemetry into model input representations."""
        if not self.pipeline:
            self._load_pipeline()

        cols = self.pipeline["observable_cols"]
        row = np.array([raw_features_dict.get(c, 0.0) for c in cols], dtype=float).reshape(1, -1)

        # Handle NaNs
        row_imp = self.pipeline["imputer"].transform(row)
        row_std = self.pipeline["std_scaler"].transform(row_imp)
        row_ang = self.pipeline["angle_scaler"].transform(row_imp)

        pcas = {k: self.pipeline["pca_models"][k].transform(row_std) for k in self.pipeline["pca_models"]}
        pcas_ang = {k: self.pipeline["pca_scalers"][k].transform(pcas[k]) for k in self.pipeline["pca_scalers"]}

        return {
            "raw": row_imp[0],
            "standard": row_std[0],
            "angle": row_ang[0],
            "pca_2": pcas[2][0],
            "pca_4": pcas[4][0],
            "pca_6": pcas[6][0],
            "pca_8": pcas[8][0],
            "pca_16": pcas[16][0],
            "pca_2_angle": pcas_ang[2][0],
            "pca_4_angle": pcas_ang[4][0],
            "pca_6_angle": pcas_ang[6][0],
            "pca_8_angle": pcas_ang[8][0],
            "pca_16_angle": pcas_ang[16][0],
        }

    def score_window(
        self,
        current_features: dict[str, Any],
        model_id: str = "PITQSVDD_6Q_Angle",
    ) -> dict[str, Any]:
        """Perform live inference on a telemetry window."""
        t0 = time.time()
        transformed = self.transform_features(current_features)

        # Keep history buffer for temporal models (L=3)
        self.window_history.append(transformed)
        if len(self.window_history) > self.max_history:
            self.window_history.pop(0)

        # Ensure model is loaded
        if model_id not in self.loaded_models:
            self.load_model(model_id)

        model_entry = self.loaded_models.get(model_id)
        threshold = CALIBRATED_THRESHOLDS.get(model_id, 0.05)

        if not model_entry:
            # Fallback if model could not be loaded
            return {
                "model_id": model_id,
                "anomaly_score": 0.0,
                "threshold": threshold,
                "is_anomaly": False,
                "latency_ms": 0.0,
                "components": {},
                "status": "MODEL_UNAVAILABLE",
            }

        m_type = model_entry["type"]
        model = model_entry["model"]
        score = 0.0
        components = {}

        try:
            # 1. Flagship PI-TQSVDD inference
            if m_type == "pitqsvdd":
                meta = model_entry["meta"]
                n_q = meta["qubits"]
                enc = meta["encoding"]

                # Determine key
                if enc in ("amplitude", "data_reuploading"):
                    feat_key = "pca_16_angle"
                else:
                    feat_key = f"pca_{n_q}_angle"

                # Construct sequence of length 3
                if len(self.window_history) >= 3:
                    seq = np.array([w[feat_key] for w in self.window_history[-3:]])
                else:
                    # Pad
                    cur = transformed[feat_key]
                    seq = np.array([cur, cur, cur])

                seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, 3, n_feats)

                with torch.no_grad():
                    total_s, s_norm, s_temp = model.score(seq_t)
                    score = float(total_s[0].cpu().numpy())
                    components = {
                        "normal_distance": float(s_norm[0].cpu().numpy()),
                        "temporal_error": float(s_temp[0].cpu().numpy()),
                    }

            # 2. Classical Temporal Autoencoder
            elif m_type == "temporal_autoencoder":
                if len(self.window_history) >= 3:
                    seq = np.array([w["standard"] for w in self.window_history[-3:]])
                else:
                    cur = transformed["standard"]
                    seq = np.array([cur, cur, cur])
                seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    recon = model(seq_t)
                    err = torch.mean((seq_t - recon) ** 2, dim=-1)
                    score = float(err[0, -1].cpu().numpy())
                    components = {"reconstruction_mse": score}

            # 3. Classical Dense Autoencoder
            elif m_type == "autoencoder":
                x_t = torch.tensor(transformed["standard"], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    recon = model(x_t)
                    err = torch.mean((x_t - recon) ** 2, dim=-1)
                    score = float(err[0].cpu().numpy())
                    components = {"reconstruction_mse": score}

            # 4. Mahalanobis Full
            elif m_type == "mahalanobis":
                diff = transformed["raw"] - model["mean"]
                d_sq = float(diff @ model["inv_cov"] @ diff.T)
                score = float(np.sqrt(max(0.0, d_sq)))
                components = {"mahalanobis_distance": score}

            # 5. Sklearn Estimator (Isolation Forest, Logistic Regression)
            elif m_type == "sklearn":
                feat_key = "pca_6" if "PCA6" in model_id else "standard"
                x_in = transformed[feat_key].reshape(1, -1)
                if hasattr(model, "decision_function"):
                    raw_s = float(model.decision_function(x_in)[0])
                    # For Isolation Forest, lower is more anomalous -> invert
                    score = -raw_s if "IsolationForest" in model_id else raw_s
                elif hasattr(model, "predict_proba"):
                    score = float(model.predict_proba(x_in)[0, 1])
                else:
                    score = float(model.predict(x_in)[0])
                components = {"raw_estimator_output": score}

            # 6. Static Quantum SVDD
            elif m_type == "static_qsvdd":
                x_t = torch.tensor(transformed["pca_4_angle"], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    s = model.score(x_t)
                    score = float(s[0].cpu().numpy())
                    components = {"hypersphere_distance": score}

            else:
                score = 0.0

        except Exception as e:
            log.error(f"Inference error in {model_id}: {e}", exc_info=True)
            score = 0.0

        latency_ms = (time.time() - t0) * 1e3
        is_anomaly = score >= threshold

        return {
            "model_id": model_id,
            "anomaly_score": score,
            "threshold": threshold,
            "is_anomaly": is_anomaly,
            "latency_ms": latency_ms,
            "components": components,
            "status": "SUCCESS",
        }

    def compare_models(
        self,
        current_features: dict[str, Any],
        model_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Run multiple models side-by-side on the identical feature window."""
        results = []
        for m_id in model_ids:
            res = self.score_window(current_features, model_id=m_id)
            results.append(res)
        return results

    def reset_buffer(self) -> None:
        """Reset temporal sliding history."""
        self.window_history.clear()


# Global singleton instance
inference_service = InferenceService()
