# QML Review 2 — Pre-Experimental Audit and Specification
## Quantum Machine Learning (QML) Anomaly-Detection Extension for QISM

---

### Executive Audit Metadata
- **Project:** Quantum Infrastructure Security Monitor (QISM)
- **Document:** Review 2 QML Pre-Experimental Audit (`QML_REVIEW2_AUDIT.md`)
- **Author/Investigator:** QISM Technical Review Team
- **Baseline Version:** V1.0 Hardened Release (Git commit baseline on branch `main`)
- **Objective:** Establish the dataset, feature schema, label mappings, execution environment, and experimental design for lightweight QML anomaly-detection models (VQC and Quantum Kernel SVM) to be presented during University Project Review 2.
- **Audit Date:** Current Local Baseline

---

## 1. Project Root, Git Status, and Isolation Boundary

### 1.1 Git Status and Workspace Integrity
- **Repository Root:** `c:\Users\Shriram\Desktop\python\QKD_Sim\qism-simulator`
- **Active Branch:** `main` (up to date with `origin/main`)
- **Working Tree State:** Core simulator (`src/qism/simulator/`, `src/qism/execution/`) and established V0.6–V1.0 artifacts are preserved.
- **Isolation Policy:** All QML-specific code, configuration, test scripts, and execution outputs will reside strictly in dedicated directories:
  - Code: `qml/` (subdivided into `config/`, `src/`, `tests/`)
  - Artifacts: `artifacts/qml_review2/` (subdivided into `figures/`, `metrics/`, `predictions/`, `manifests/`, `checkpoints/`)
  - Documentation: `docs/QML_REVIEW2_*.md`
  - Existing V0.6–V1.0 checkpoints (`models/checkpoints/`), logs, and historical outputs (`outputs/`, `experiments/`) will **not** be overwritten or modified.

---

## 2. Python Environment and Dependency Audit

### 2.1 Python Runtime
- **Virtual Environment Path:** `.venv\Scripts\python.exe`
- **Python Version:** `3.10.11` (AMD64 Windows)
- **Core Scientific Libraries:**
  - `numpy`: `2.2.6`
  - `scipy`: `1.15.3`
  - `pandas`: `2.2.3`
  - `scikit-learn`: `1.7.2`
  - `torch`: `2.14.0` (CPU execution)
  - `matplotlib`: `3.10.9`
  - `pyarrow`: `19.0.1`

### 2.2 Quantum Framework Selection and Verification
- **Framework Installed:** `pennylane` (`0.42.3`) alongside `pennylane-lightning` (`0.42.0`).
- **Compatibility Status:** Clean installation into `.venv` without conflicts or downgrades to PyTorch or Scikit-learn.
- **Simulator Device:** `default.qubit` (pure state vector quantum simulator, highly optimized in Python/NumPy for $n \le 10$ qubits).
- **Import and Gradient Verification:**
  - `qml.device('default.qubit', wires=4)` instantiated and verified.
  - Automatic differentiation interface: PyTorch (`qml.qnode(dev, interface='torch', diff_method='backprop')`) verified.

---

## 3. Dataset Audit: Structure, Splits, and Observability Boundaries

### 3.1 Dataset Selection
- **Selected Dataset:** `data/v0.5_small/v05_small_dataset/`
  - *Rationale:* This is the canonical benchmark dataset created for the QISM ML evaluation pipeline. It provides pre-partitioned scenarios generated with full physical simulation (BB84 decoy states, channel attenuation, and SPAD detection), making QML findings directly comparable with the existing V0.6 classical ML benchmarks.
- **Total Samples:** 100 telemetry windows (each aggregating 2,000 optical pulses) across 20 distinct scenarios.

### 3.2 Partition Structure and Sample Allocation
The dataset is partitioned via `MLDatasetLoader` and `SplitManager` (`src/qism/ml/data/`):

| Partition Name | Scenarios Count | Window Count | Class Distribution | Physical Scenarios Included |
|---|---|---|---|---|
| **`train`** | 4 | 20 | 20 Normal (100%), 0 Attack | `normal_0000`, `normal_0001`, `normal_0002`, `normal_0004` |
| **`val`** | 2 | 10 | 10 Normal (100%), 0 Attack | `normal_0003`, `normal_0005` |
| **`test` (IID)** | 8 | 40 | 18 Normal (45%), 22 Attack (55%) | `attack_0000` to `attack_0005` + `normal_0003`, `normal_0005` |
| **`generalization` (OOD)**| 4 | 20 | 8 Normal (40%), 12 Attack (60%) | `generalization_0000` to `generalization_0003` |
| **`calibration`** | 4 | 20 | 20 Normal (100%), 0 Attack | `calibration_0000` to `calibration_0003` |

### 3.3 Target Label Definition
- **Label Column:** `ground_truth_attack_active`
  - $y = 0$: Normal operation (legitimate BB84 decoy-state QKD).
  - $y = 1$: Attack active (eavesdropping actively underway).
- **Attack Types Present in Test Set:**
  - `source_manipulation` (15 windows)
  - `pns` (Photon-Number Splitting, 4 windows)
  - `detector_manipulation` (blinding/control, 3 windows)
- **Positive Class:** Class 1 (`Attack`). All precision, recall, and F1 metrics treat `Attack` as the positive detection class.

### 3.4 Observability Boundary and Anti-Leakage Audit
- In accordance with `src/qism/ml/features/definitions.py`, all columns in `GROUND_TRUTH_FIELDS` (`ground_truth_attack_active`, `ground_truth_attack_type`, `ground_truth_attack_fraction`, etc.) and `METADATA_FIELDS` (`run_id`, `scenario_id`, `window_id`, `timestamp`) are strictly excluded from the input feature space.
- Preprocessing transformations (`StandardScaler`, PCA) will be fitted **exclusively** on `train` normal samples and applied to `val`, `test`, and `generalization`.

---

## 4. Dimensionality Reduction & Quantum Feature Representation

### 4.1 Quantum Computational Constraint
- Quantum circuits simulated on classical hardware scale as $\mathcal{O}(2^n)$. To achieve rapid, stable, and reproducible training without stochastic barren plateau issues, the quantum circuit will use **$n = 4$ qubits**.

### 4.2 Compact Feature Strategies
We establish two parallel 4-dimensional representations derived from the raw telemetry:

1. **Representation A (Unsupervised PCA - 4 Components):**
   - Fitted strictly on `X_train` ($N=20, d=48$ features).
   - Retains $>85\%$ of the total variance across optical, detector, and timing dynamics.
   - Scaled using Min-Max scaling to the interval $[0, \pi]$ for angle embedding.
2. **Representation B (Physical QKD Domain Telemetry - 4 Features):**
   - Four physically grounded, observable telemetry metrics:
     1. `qber`: Quantum Bit Error Rate (polarization error).
     2. `sifted_rate`: Sifted key generation rate.
     3. `signal_gain`: Detection probability of signal pulses.
     4. `detector_imbalance`: Relative click rate discrepancy between detector 0 and detector 1.
   - Scaled using training min/max to $[0, \pi]$.

---

## 5. Existing Classical ML Baselines for Direct Benchmarking

From existing V0.6 artifacts on `v05_small_dataset` (`experiments/` and `docs/QISM_ML_BENCHMARK_RESULTS.md`):

| Model Name | Type | Feature Space | Test Precision | Test Recall | Test F1 | Test AUROC | Test AUPRC |
|---|---|---|---|---|---|---|---|
| **Robust MCD (Elliptic)** | Classical Unsupervised | Full (48 feats) | 0.6923 | 0.8182 | 0.7500 | 0.7854 | 0.8090 |
| **Dense Autoencoder** | Deep Learning Unsupervised | Full (48 feats) | 0.7037 | 0.8636 | 0.7755 | 0.6793 | 0.6473 |
| **Isolation Forest** | Classical Tree Ensemble | Full (48 feats) | 0.6667 | 0.0909 | 0.1600 | 0.6237 | 0.6468 |

### Direct Fair Classical Baselines (Trained on the Exact Same 4-Feature Inputs):
To provide a scientifically rigorous, apples-to-apples comparison on the exact same 4-dimensional representation:
1. **Classical Logistic Regression:** Linear reference baseline with L2 regularization.
2. **Classical Support Vector Classifier (RBF-SVM):** Direct non-linear reference for the Quantum Kernel SVM.
3. **Classical Lightweight MLP (4 $\to$ 8 $\to$ 1):** Direct classical neural reference for the Variational Quantum Classifier (comparable parameter count $\approx 45$ params).

---

## 6. Proposed QML Model Architectures

### 6.1 Model 1: Hybrid Variational Quantum Classifier (VQC)
- **Qubits:** 4 qubits ($q_0, q_1, q_2, q_3$).
- **State Preparation:** Angle Embedding $U_{\text{enc}}(\mathbf{x}) = \bigotimes_{i=0}^3 R_Y(x_i) |0\rangle^{\otimes 4}$.
- **Variational Circuit:** 2 layers of `StronglyEntanglingLayers` (or parameterized $R_X, R_Y, R_Z$ rotations on each wire followed by CNOT ring entanglement).
  - Total variational parameters: $2 \text{ layers} \times 4 \text{ wires} \times 3 \text{ angles} = 24$ trainable parameters.
- **Measurement:** Expectation values of Pauli-$Z$ on all 4 qubits: $z_i = \langle \psi(\mathbf{x}, \boldsymbol{\theta}) | Z_i | \psi(\mathbf{x}, \boldsymbol{\theta}) \rangle \in [-1, 1]$.
- **Classical Post-Processing:** Trainable linear layer $w \in \mathbb{R}^4, b \in \mathbb{R}$ mapped via sigmoid:
  $$p(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{z} + b)$$
- **Optimization:** PyTorch Adam optimizer ($\eta = 0.02$, $\beta_1 = 0.9, \beta_2 = 0.999$), Binary Cross-Entropy loss ($\mathcal{L}_{\text{BCE}}$), 30 epochs with training/validation loss logging.

### 6.2 Model 2: Quantum Kernel Support Vector Machine (QSVM)
- **Qubits:** 4 qubits.
- **Quantum Feature Map:** Angle embedding + entangling CNOT layer:
  $$|\phi(\mathbf{x})\rangle = U_{\text{map}}(\mathbf{x})|0\rangle^{\otimes 4} = \text{CNOT}_{\text{ring}} \left( \bigotimes_{i=0}^3 R_Y(x_i) \right) |0\rangle^{\otimes 4}$$
- **Fidelity Quantum Kernel:**
  $$K(\mathbf{x}, \mathbf{x}') = |\langle \phi(\mathbf{x}) | \phi(\mathbf{x}') \rangle|^2$$
  Evaluated via projective measurement on the state $|0\rangle^{\otimes 4}$ after applying $U_{\text{map}}(\mathbf{x}) U_{\text{map}}^\dagger(\mathbf{x}')$.
- **Classical SVM Solver:** Scikit-learn `SVC(kernel='precomputed')`.
  - Kernel matrix cached: Train ($20 \times 20$), Val ($10 \times 20$), Test ($40 \times 20$).
  - Regularization parameter $C$ selected via validation performance.

### 6.3 Model 3 (Optional): Lightweight Quantum Autoencoder (QAE)
- To be evaluated strictly after Models 1 and 2 are fully completed and documented.
- Architecture: 4 qubits (2 latent, 2 trash), trained on normal samples using a swap-test fidelity objective with auxiliary reference state $|0\rangle$.

---

## 7. Audit Sign-Off
- **Dataset Ready:** `data/v0.5_small/v05_small_dataset/` verified and accessible.
- **Environment Verified:** PennyLane 0.42.3 functional on Python 3.10.11.
- **Anti-Leakage Protocol Enforced:** Zero label contamination, train-only scalers.
- **Execution Authorized:** Proceeding to Phase B (Minimal working circuits and pilot tests).

---
