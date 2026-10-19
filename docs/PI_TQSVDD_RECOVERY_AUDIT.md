# QISM PI-TQSVDD — Phase 0: Repository Recovery and Baseline Audit
## Physics-Informed Temporal Quantum Support Vector Data Description for QKD Attack Anomaly Detection

---

### Executive Audit Metadata
- **Project:** Quantum Infrastructure Security Monitor (QISM)
- **Document:** Phase 0 Recovery and Baseline Audit (`docs/PI_TQSVDD_RECOVERY_AUDIT.md`)
- **Investigator / Architect:** Antigravity AI Senior Research Assistant
- **Baseline Version:** QISM V1.0 Hardened Release (Git Commit: `869f6cd`, Branch: `main`)
- **Primary Framework:** PennyLane 0.42.3 with `pennylane-lightning` 0.42.0
- **Quantum Simulator Backend:** `lightning.qubit` (Verified with adjoint differentiation and PyTorch 2.14.0)
- **Required Qubit Configurations:** 2, 4, 6, and 8 qubits
- **Audit Date:** Current Baseline (October 2026)

---

## 1. Repository Inventory and Working Tree Baseline

### 1.1 Git Status & Integrity
- **Repository Root:** `c:\Users\Shriram\Desktop\python\QKD_Sim\qism-simulator`
- **Active Branch:** `main`
- **Latest Commit:** `869f6cd` (*feat(v0.6): complete QISM simulator and anomaly detection ML benchmarking infrastructure*)
- **Working Tree State:**
  - Modified tracked files: `README.md`, `src/qism/cli.py`, `src/qism/random.py`
  - Untracked components from Review 2:
    - Code: `qml/src/` (VQC, QSVM, QAE prototypes), `qml/tests/`
    - Artifacts: `artifacts/qml_review2/` (checkpoints, figures, manifests, metrics, predictions)
    - Documentation: `docs/QML_REVIEW2_AUDIT.md`, `docs/QML_REVIEW2_EXPERIMENT_REPORT.md`, `docs/QML_REVIEW2_PRESENTATION_NOTES.md`
    - Classical and V0.6–V1.0 artifacts: `docs/QISM_V0_6_TO_V1_0_COMPREHENSIVE_TECHNICAL_REPORT.md`, `docs/QISM_ML_BENCHMARK_RESULTS.md`
- **Preservation Policy:** No existing simulator files (`src/qism/simulator/`), classical ML pipelines (`src/qism/ml/`), or historical Review 2 artifacts (`artifacts/qml_review2/`) will be overwritten, modified, or deleted. All new PI-TQSVDD components will be housed in isolated modules.

### 1.2 Python Environment and Dependency Audit
- **Python Binary:** `c:\Users\Shriram\Desktop\python\QKD_Sim\qism-simulator\.venv\Scripts\python.exe`
- **Python Version:** `3.10.11` (64-bit AMD64 Windows)
- **Core Scientific Libraries:**
  - `numpy`: `2.2.6`
  - `scipy`: `1.15.3`
  - `pandas`: `2.2.3`
  - `scikit-learn`: `1.7.2`
  - `torch`: `2.14.0` (CPU execution)
  - `pyarrow`: `19.0.1`
  - `matplotlib`: `3.10.9`
- **Quantum Framework & Simulator:**
  - `pennylane`: `0.42.3`
  - `pennylane-lightning`: `0.42.0`
  - **Required Simulator Backend:** `lightning.qubit` verified and tested.
  - **Differentiation Interface:** PyTorch automatic differentiation via `diff_method="adjoint"`. Verified with test execution and gradient backpropagation (`w.grad.norm() > 0`).

---

## 2. Dataset and Telemetry Schema Audit

### 2.1 Available Datasets
The project contains several datasets in `data/`:
1. `data/v0.5_small/v05_small_dataset/` (**Canonical ML Benchmark Dataset**):
   - 20 scenarios, 100 aggregated telemetry windows (2,000 pulses per window).
   - Contains scenario manifest: `scenarios/scenario_manifest.json` with deterministic splits (`calibration`: 4, `normal`: 6, `attack`: 6, `generalization`: 4).
   - Telemetry parquet files containing 130 columns per scenario window.
2. `data/v0.5_smoke/` and `data/v04_experiments/`: Diagnostic and ablation sets.

### 2.2 Telemetry Feature Schema & Observability Boundary
- **Raw Scenario Telemetry:** 130 columns.
- **Strictly Observable Feature Space (`FeatureExtractor(feature_set='full')`):** 48 continuous telemetry columns grouped into 4 physical families:
  1. *QBER & Optical Error Statistics* (10 features)
  2. *Detector Dynamics & Coincidence Statistics* (13 features)
  3. *Environmental & Hardware Sensors* (13 features)
  4. *Timing & Channel Dispersion* (12 features)
- **Strictly Excluded Ground Truth & Metadata:**
  - `ground_truth_attack_active`, `ground_truth_attack_type`, `ground_truth_attack_fraction`, `ground_truth_mean_severity`, `ground_truth_attack_id`
  - `scenario_id`, `run_id`, `window_id`, `timestamp`, `pulse_id`
  - Unobservable internal simulator state variables (`true_temperature_c`, `photon_number`, `mean_channel_transmittance`, etc.)
- **Deficiency of Previous Review 2 Approach:**
  - Previous Review 2 relied exclusively on either a 4-component PCA or 4 hand-picked features (`qber`, `sifted_rate`, `signal_gain`, `detector_imbalance`).
  - This 4-feature bottleneck discarded over 91% of the available telemetry channels, potentially ignoring subtle attacks like low-intensity timing jitter or decoy-state yield shifts.
  - PI-TQSVDD explicitly requires investigating mechanisms to consume the full 48-feature representation through learned classical projections, feature-group partitioning, and multi-layer data reuploading.

---

## 3. Review of Existing QML Implementations

| Component | Location | Model Formulation | Status / Limitations |
|---|---|---|---|
| **Supervised VQC** | `qml/src/vqc_model.py` | 4 qubits, AngleEmbedding ($R_Y$), StronglyEntanglingLayers ($L=2$), Pauli-$Z$ expval $\to$ Linear head $\to \sigma$ | Supervised only ($L_{\text{BCE}}$ on attack labels). Not one-class. Requires attack labels for training. |
| **Quantum Kernel SVM** | `qml/src/quantum_kernel_svm.py` | 4 qubits, AngleEmbedding + CNOT ring feature map, state fidelity Gram matrix $\to$ Scikit-Learn SVC | Supervised or binary. Computational cost $\mathcal{O}(N^2)$ quantum evaluations. Does not scale smoothly to temporal sequences. |
| **Quantum Autoencoder (QAE)** | `qml/src/quantum_autoencoder.py` | 4 qubits (2 latent, 2 trash), AngleEmbedding + StronglyEntanglingLayers, trash fidelity with $\|00\rangle$ | Unsupervised, but limited to static single windows; does not model temporal dynamics or state progression. |
| **Classical Baselines** | `qml/src/baselines.py` | Logistic Regression, RBF-SVM, Small MLP, Isolation Forest | Trained on 4 features. Missing classical Deep SVDD and classical Temporal Autoencoder baselines. |

### 3.1 Gaps Addressed by PI-TQSVDD
1. **One-Class Normality Objective:** Legitimate normal-only training objective using quantum representation distance to hypersphere centre $c$.
2. **Temporal Modeling:** Causal temporal prediction objective $\mathcal{L}_{\text{temporal}} = \|z_t - \hat{z}_t\|_2^2$ over quantum latent states to capture physical attack trajectories.
3. **Multi-Qubit Scaling:** Rigorous evaluation across 2, 4, 6, and 8 qubits using `lightning.qubit`.
4. **Encoding Benchmark:** Systematic comparison of basic angle, data reuploading, learned bottleneck projections, feature grouping, IQP/ZZ, and amplitude encoding.
5. **Full Telemetry Utilization:** Preserving information across all 48 telemetry features rather than naive 4-feature truncation.
6. **Robust Resumability:** Crash-resistant atomic checkpointing supporting automated pause and resume from any step without restarting from epoch zero.

---

## 4. Isolation and Architecture Plan

### 4.1 Directory Structure
To maintain strict isolation from existing V0.6–V1.0 and Review 2 code:
- **Core Package:** `qml/pitqsvdd/`
  - `config.py`: Dataclasses and YAML configuration loader.
  - `data.py`: Normal-only training partitions, temporal windowing, anti-leakage scaling.
  - `feature_transforms.py`: Full 48-feature transformations (PCA, learned bottleneck, feature grouping, random projections).
  - `temporal_windows.py`: Causal sequence construction without boundary leakage.
  - `encodings/`: Modular encoding classes (Angle, Data Reuploading, Amplitude, IQP/ZZ, Feature-Group, Learned Projection).
  - `circuits/`: Variational quantum circuits with configurable depth, entanglement, and measurement observables using `lightning.qubit`.
  - `models/`: Static QSVDD, Temporal PI-TQSVDD, and Context-Aware PI-TQSVDD.
  - `training/`: Resumable trainer, atomic checkpoint manager, experiment runner.
  - `evaluation/`: One-class anomaly scoring, quantile threshold calibration, point/event metrics.
  - `visualization/`: Automated publication-grade plotting utilities.
  - `tests/`: Automated unit and integration test suite.
- **Experiment Artifacts:** `artifacts/pi_tqsvdd/`
  - `checkpoints/`: Resumable checkpoint files (`.pt`).
  - `figures/`: Diagnostic and evaluation plots.
  - `metrics/`: JSON performance metrics for every run.
  - `manifests/`: Run manifests with full provenance metadata.
  - `experiment_registry.csv` and `experiment_registry.json`: Machine-readable experiment records.
- **Documentation Deliverables:** `docs/PI_TQSVDD_*.md`

---

## 5. Computational Risk and Mitigation Strategy

1. **Exponential Simulation Scaling:**
   - Quantum state vector simulation scales as $\mathcal{O}(2^n)$. An 8-qubit circuit has state dimension $2^8 = 256$, which is well within CPU RAM limits, but gradient evaluations across many epochs can be time-consuming.
   - *Mitigation:* Use `lightning.qubit` with C++ backend and `diff_method="adjoint"`. Conduct budget-controlled pilot runs (Mode B) with small normal training sets before preparing the final training configuration (Mode C).
2. **Representation Collapse in Deep SVDD:**
   - In neural and quantum SVDD, setting all weights to zero or driving representations to a constant point trivially minimizes $\|z - c\|_2^2 \to 0$.
   - *Mitigation:* Remove bias terms in the final quantum layer, initialize weights away from zero with random orthogonal or normal initialization, compute centre $c$ via an initial forward pass on normal data, and fix $c$ throughout training.
3. **Temporal Leakage Across Windows:**
   - Overlapping rolling windows spanning across scenario boundaries will contaminate training and evaluation.
   - *Mitigation:* Partition scenarios first, then construct causal temporal sequences strictly within each scenario.
4. **Process Interruptions and Crashes:**
   - *Mitigation:* Implement two-stage atomic checkpointing (write to `.tmp`, flush to disk, atomic rename/replace) and an automated test verifying deterministic resume.

---

## 6. Audit Sign-Off
- **Repository Baseline:** Verified and preserved.
- **Backend Verified:** PennyLane `lightning.qubit` functional with PyTorch adjoint gradients.
- **Telemetry Schema:** 130 columns identified; 48 observable features classified into 4 physical families.
- **Execution Authorized:** Proceeding to Phase 1 (Full Telemetry Feature Audit).
