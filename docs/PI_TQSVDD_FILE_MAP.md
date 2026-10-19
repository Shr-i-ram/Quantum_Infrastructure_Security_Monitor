# QISM PI-TQSVDD — Definitive Project File Map
## Complete Path Inventory, Architectural Roles, and Modification Guidelines

---

### Executive File Map Metadata
- **Project:** Quantum Infrastructure Security Monitor (QISM)
- **Document:** Definitive Project File Map (`docs/PI_TQSVDD_FILE_MAP.md`)
- **Repository Root:** `c:\Users\Shriram\Desktop\python\QKD_Sim\qism-simulator`
- **Audit Date:** Current Baseline (October 2026)

---

## 1. Essential Files (Core Architecture & Execution)

These are the primary files required to understand, configure, execute, and evaluate the PI-TQSVDD flagship model.

| File Path | Functional Role | Should You Edit? |
|---|---|:---:|
| `qml/pitqsvdd/models/temporal_pitqsvdd.py` | **Flagship Model Architecture:** Defines `TemporalPITQSVDD`, combining quantum hypersphere distance with causal GRU latent residual loss. | **No:** Core mathematical formulation; fully verified. |
| `qml/pitqsvdd/circuits/variational.py` | **Quantum Circuit Implementation:** Implements `VariationalQuantumCircuit` on PennyLane `lightning.qubit` with adjoint automatic differentiation. | **No:** Verified across 2, 4, 6, 8 qubits. |
| `qml/pitqsvdd/encodings/data_reuploading.py` | **Flagship Quantum Encoding:** Implements multi-layer data reuploading distributing high-dimensional features across layers. | **No:** Winner encoding for final configuration. |
| `qml/pitqsvdd/encodings/` | **Modular Quantum Encodings:** Implementations for `angle.py`, `amplitude.py`, `iqp_zz.py`, `feature_group.py`, and `learned_projection.py`. | **No:** Fully implemented and benchmarked. |
| `qml/pitqsvdd/feature_transforms.py` | **Feature Preprocessing:** Fits imputation, standardization, and PCA strictly on normal training data without leakage. | **No:** Preserves 48 physical telemetry channels. |
| `qml/pitqsvdd/temporal_windows.py` | **Temporal Window Builder:** Constructs causal sequences strictly within individual scenarios, preventing cross-scenario boundary leakage. | **No:** Fully verified by unit tests. |
| `qml/pitqsvdd/train.py` | **Command-Line Entry Point:** Dedicated training CLI accepting `--config`, `--run-id`, `--resume`, `--qubits`, `--encoding`. | **No:** Standard executable entry point. |
| `qml/pitqsvdd/configs/pitqsvdd_final.yaml` | **Final Training Configuration:** Definitive configuration for the 4-qubit data reuploading flagship run (30 epochs, Adam lr=0.01). | **Optional:** Edit only if tuning batch size or epochs. |
| `qml/pitqsvdd/training/checkpointing.py` | **Crash-Resilient Checkpoint Manager:** Implements two-stage atomic file replacement (`checkpoint_latest.pt`, `checkpoint_best.pt`). | **No:** Fully verified and tested. |
| `qml/pitqsvdd/training/resume.py` | **Resume Resolution Logic:** Guardrails auto-resumption and checks overwrite flags. | **No:** Fixed during Phase C audit. |
| `qml/pitqsvdd/evaluation/scoring.py` | **Evaluation Entry Point:** Computes combined anomaly scores and calibrates decision thresholds on normal validation data. | **No:** Prevents test label leakage. |
| `qml/pitqsvdd/evaluation/metrics.py` | **Point-Level Metrics:** Computes Precision, Recall, F1, Specificity, AUROC, AUPRC, FPR, FNR, and confusion matrices. | **No:** Standard scientific evaluation. |
| `qml/pitqsvdd/evaluation/event_metrics.py` | **Event-Level Metrics:** Computes incident recall, mean detection delay in windows, and false alarm rate. | **No:** Operational metrics engine. |
| `qml/pitqsvdd/benchmark_runner.py` | **Automated Pilot Benchmark Runner:** Orchestrates the multi-qubit and multi-encoding pilot experiments and classical comparisons. | **No:** Automated benchmark script. |
| `docs/PI_TQSVDD_HANDOVER_GUIDE.md` | **Lead Researcher Handover Guide:** Step-by-step instructions, mathematical formulation, and exact terminal commands. | **Reference Document.** |
| `docs/PI_TQSVDD_FINAL_AUDIT.md` | **Final Implementation Audit Report:** Complete requirement verification matrix and audit findings. | **Reference Document.** |

---

## 2. Supporting Files (Configs, Artifacts, Figures & Tests)

### 2.1 Configuration Files
- `qml/pitqsvdd/configs/pitqsvdd_final.yaml`: Definitive final training configuration (Mode C).
- `qml/pitqsvdd/configs/smoke.yaml`: Fast 2-epoch smoke test configuration (Mode A).
- `pyproject.toml`: Root project configuration specifying package metadata, dependencies, and `pythonpath = [".", "src"]`.

### 2.2 Experiment Registries and Manifests
- `artifacts/pi_tqsvdd/experiment_registry.csv`: Machine-readable master experiment table recording all runs, parameters, and metrics.
- `artifacts/pi_tqsvdd/experiment_registry.json`: JSON format of the experiment registry.
- `artifacts/pi_tqsvdd/classical_baselines_summary.json`: Benchmark results for Classical Deep SVDD, Isolation Forest, and One-Class SVM.
- `artifacts/pi_tqsvdd/manifests/*.json`: Provenance metadata and configuration dumps for each completed experiment run.

### 2.3 Model Checkpoints
- `artifacts/pi_tqsvdd/checkpoints/*/checkpoint_latest.pt`: Most recent completed epoch checkpoint.
- `artifacts/pi_tqsvdd/checkpoints/*/checkpoint_best.pt`: Best validation loss checkpoint.
- `artifacts/pi_tqsvdd/checkpoints/*/checkpoint_epoch_{N}.pt`: Periodic archival checkpoint.

### 2.4 Performance Metrics & Figures
- `artifacts/pi_tqsvdd/metrics/*.json`: Detailed evaluation outputs, score distributions, and point/event metrics for each run.
- `artifacts/pi_tqsvdd/figures/01_pitqsvdd_architecture_overview.png`: Architectural data flow schematic.
- `artifacts/pi_tqsvdd/figures/02_telemetry_features_and_pca_variance.png`: Feature taxonomy and PCA explained variance.
- `artifacts/pi_tqsvdd/figures/03_qubits_and_encodings_benchmark.png`: Multi-panel qubit count, encoding, and wall-time benchmark.
- `artifacts/pi_tqsvdd/figures/04_flagship_loss_convergence.png`: Loss convergence and SVDD/temporal decomposition curves.
- `artifacts/pi_tqsvdd/figures/05_anomaly_score_distribution.png`: Anomaly score separation (Normal vs Attack).
- `artifacts/pi_tqsvdd/figures/06_roc_pr_curves_flagship.png`: Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.
- `artifacts/pi_tqsvdd/figures/07_model_family_comparison.png`: Grouped bar chart comparing PI-TQSVDD with classical baselines.

### 2.5 Automated Test Suite
- `qml/pitqsvdd/tests/test_checkpoint_resume.py`: Verifies crash-resistant serialization, restart, and state continuation.
- `qml/pitqsvdd/tests/test_circuits.py`: Verifies forward and adjoint backward passes across 2, 4, 6, 8 qubits.
- `qml/pitqsvdd/tests/test_data.py`: Verifies zero attack contamination, causal sliding windows, and anti-leakage scaling.
- `qml/pitqsvdd/tests/test_encodings.py`: Verifies all 6 quantum encoding implementations.
- `qml/pitqsvdd/tests/test_losses.py`: Verifies SVDD hypersphere, temporal residual, and multi-centre loss functions.
- `qml/pitqsvdd/tests/test_metrics.py`: Verifies point metrics, threshold calibration, and event delay calculations.

---

## 3. Files That Should NOT Be Modified

The following components represent completed foundation modules of the QISM platform. They must not be modified or deleted during QML experiments:

1. **Simulator Core Engine (`src/qism/simulator/`):**
   - Implements discrete-variable decoy-state BB84 physics, optical channel attenuation, SPAD detector state machines, and physical attack injections (V0.1–V0.4).
2. **Canonical Datasets (`data/v0.5_small/`):**
   - The primary multi-scenario parquet telemetry files generated under Latin Hypercube parameter distributions (V0.5).
3. **Classical ML Benchmarking Infrastructure (`src/qism/ml/`):**
   - Classical feature extractors, loaders, and baseline detectors (Isolation Forest, Robust MCD, Dense Autoencoder) from V0.6–V0.7.
4. **Drift & Monitoring Infrastructure (`src/qism/drift/`, `src/qism/monitor/`):**
   - Streaming circular ring buffers, concept drift detectors, and alarm state machines from V0.8–V1.0.
5. **Review 2 Historical Prototypes (`qml/src/`, `artifacts/qml_review2/`):**
   - Historical supervised VQC and Quantum Kernel SVM prototypes preserved for comparative auditing.
