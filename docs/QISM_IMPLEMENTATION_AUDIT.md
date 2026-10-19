# Quantum Infrastructure Security Monitor (QISM)
## Comprehensive Research Benchmark Implementation Audit and Scientific Evidence Report

**Auditor Role:** Senior Research Software Auditor (Quantum Machine Learning, Anomaly Detection, Experimental Methodology, Scientific Computing)  
**Date of Audit:** October 20, 2026  
**Audit Target:** QISM 76-Observable Feature Research Benchmark (`data/qism_research_dataset`)  
**Repository Version:** Simulator 0.6.0 | Dataset Schema 0.5.0 | Configuration Hash `80189033257fdeedf018b3afac859e9ffe422b2fc5019b05c899674d5eeb8499`  
**Execution Policy:** Read-only, non-destructive audit. Zero source code modifications, zero model retraining, zero table overwrite.

---

## 1. Executive Summary

This audit establishes the rigorous, evidence-based software reality of the Quantum Infrastructure Security Monitor (QISM) research benchmark. The audit evaluated all components connecting physical telemetry generation, data partitioning, preprocessing pipelines, model architectures, loss formulations, scoring policies, and persisted artifact tables.

### 1.1 Core Audit Findings
1. **Authoritative Dataset & Partition Reconciliation (VERIFIED):**
   The benchmark dataset comprises 10,100 raw telemetry windows across 505 independent simulation runs and 200 disjoint scenarios. The apparent conflict between `QISM_EXPERIMENT_PROTOCOL.md` (2,700 ID test windows) and `QISM_BENCHMARK_RESULTS.md` (2,610 ID test windows) is fully reconciled: 2,700 represents raw window counts, whereas 2,610 represents length-3 causal sliding sequences ($L=3$), which drop exactly 2 warm-up windows per scenario ($45 \times 2 = 90$ windows) to prevent temporal leakage across scenario boundaries. Because attacks never start during the first 2 windows of any scenario, 100% of the 834 attack windows are preserved.
2. **Supervised Baseline Training Flow (DISCREPANCY / LEAKAGE IDENTIFIED):**
   Unsupervised and one-class models are trained strictly on normal data from `train` (3,600 windows) with zero label access. However, all classical and quantum supervised baselines (`ClassicalLogisticRegression`, `RandomForestClassifier`, `SmallClassicalMLP`, `ClassicalRBFSVM`, `VQC_4Q`, `QuantumKernelSVM_4Q`) were trained using the 174 attack windows from `validation` combined with a normal subset from `train`. Threshold calibration for these supervised models was then performed on the normal windows of that same `validation` partition. While the test partitions remained held-out, the validation partition was compromised as an independent validation set for supervised models.
3. **Model Count Reconciliation (DISCREPANCY RECONCILED):**
   The documentation mentions 28, 29, and 30 models. Code inspection of `scripts/run_complete_benchmark.py` reveals that 31 configurations were scheduled (9 classical anomaly detectors, 4 supervised baselines, 4 quantum baselines, and 14 PI-TQSVDD ablation/seed variants). One model (`PITQSVDD_4Q_ContextAware`) crashed due to an `AttributeError` (`init_centre` vs `init_centres`), leaving exactly 30 evaluated models in `aggregate_metrics.csv` and `per_run_metrics.csv`.
4. **Quantum Kernel SVM Scale Limitation (DISCREPANCY / METHODOLOGICAL CAVEAT):**
   `QuantumKernelSVM_4Q` ranks #2 on the master leaderboard (In-Dist AUROC 0.7570). However, code inspection reveals it was evaluated on only 150 test windows (`max_samples=150`) due to quadratic Gram matrix scaling, covering only 2 of the 6 attack types (`pns` and `channel_loss`). On the stress partition, its AUROC collapsed to 0.2980. Its leaderboard ranking is not comparable to full-dataset models.
5. **PI-TQSVDD Mathematical Formulation & "Physics-Informed" Assessment (VERIFIED WITH CLARIFICATION):**
   The implemented loss function is strictly $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SVDD}} + \lambda_t \mathcal{L}_{\text{temporal}} + \lambda_{\text{reg}} \mathcal{R}$. The model contains **no explicit physics-residual loss terms, optical Hamiltonian constraints, or analytical decoy-state yield penalties**. Physical knowledge enters strictly through the 76 engineered physical telemetry features (e.g., QBER, SPAD dead-time losses, interarrival moments) and temporal continuity regularization.
6. **Statistical Validity Claims (UNSUPPORTED CLAIM IN DOCUMENTATION):**
   The benchmark report claims temporal modeling improvements are significant at $p < 0.01$ across scenario-level bootstrap resamples. Source inspection reveals that no bootstrap resampling, paired testing, or p-value computation code exists in the benchmark runner. The claim is unverified by code.
7. **Empirical Advantage Reality (SCIENTIFIC REALITY):**
   The classical temporal autoencoder (`ClassicalTemporalAutoencoder`) remains the undisputed benchmark leader (In-Dist AUROC 0.7616, Gen AUROC 0.8355, inference latency $3.7\,\mu\text{s}$ per window). While flagship `PITQSVDD_4Q_Amplitude` performs competitively with classical static autoencoders (0.7147 vs 0.7127), classical models train $100\times$ faster and perform inference $2,000\times$ to $8,000\times$ faster than simulated quantum circuits.

---

## 2. Source-Code Map

The following table provides the definitive mapping of all major components in the authoritative QISM research benchmark:

| Component | Source File | Key Symbol | What It Actually Does | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset Loader** | `scripts/run_complete_benchmark.py` | `QISMBenchmarkRunner.load_data` (L172–L211) | Discovers all `telemetry.parquet` files across 5 partitions; parses metadata (`scenario_id`, `run_id`, `window_id`, `ground_truth_*`); asserts 0% attacks in `train`. | **Verified** |
| **Feature Schema** | `src/qism/telemetry/features.py` | `get_observable_feature_names` (L571–L623) | Returns the exact 76 observable continuous physical telemetry feature names, strictly excluding all 18 ground truth columns and identifiers. | **Verified** |
| **Preprocessing & Scalers** | `scripts/run_complete_benchmark.py` | `QISMBenchmarkRunner.fit_preprocessors` (L212–L294) | Fills all-NaN columns with 0.0; fits `SimpleImputer(median)`, `StandardScaler`, `RobustScaler`, `MinMaxScaler(0, pi)`, and 5 PCA models (2, 4, 6, 8, 16) strictly on normal `train`. | **Verified** |
| **Temporal Windowing** | `scripts/run_complete_benchmark.py` | `QISMBenchmarkRunner._build_temporal_windows` (L295–L373) | Constructs causal sequence tensors $(B, L=3, D)$ strictly within each scenario; sets `has_target` boolean for next-step dynamical targets; drops first 2 warm-up windows per scenario. | **Verified** |
| **Threshold Calibration** | `scripts/run_complete_benchmark.py` | `find_threshold_at_fpr` (L120–L123) | Calibrates decision threshold on normal validation scores: $\tau = \text{percentile}(S_{\text{val\_norm}}, 100 \times (1 - \text{FPR}))$. Freezes $\tau$ before test evaluation. | **Verified** |
| **Classical Anomaly Detectors** | `scripts/run_complete_benchmark.py` | `QISMBenchmarkRunner.run_classical_baselines` (L496–L667) | Fits Isolation Forest (Full & PCA-6), One-Class SVM, LOF, regularized Mahalanobis ($76\times 76$), Robust Covariance (MCD), Dense AE, Temporal GRU-AE, and Deep SVDD on normal `train`. | **Verified** |
| **Supervised Baselines** | `scripts/run_complete_benchmark.py` | `QISMBenchmarkRunner.run_classical_supervised_baselines` (L668–L717) | Trains Logistic Regression, Random Forest, Small MLP, and RBF-SVM using 500 normal samples from `train` and 174 attack samples from `validation`. | **Verified (Leakage noted)** |
| **Quantum Baselines** | `scripts/run_complete_benchmark.py` | `QISMBenchmarkRunner.run_quantum_baselines` (L718–L824) | Trains VQC, Quantum Kernel SVM, Quantum Autoencoder, and Static Quantum SVDD on PennyLane `lightning.qubit`. | **Verified** |
| **Variational Circuit** | `qml/pitqsvdd/circuits/variational.py` | `VariationalQuantumCircuit` (L16–L148) | Executes parameterized quantum circuits with RX-RY-RZ rotations and CNOT entangling layers on PennyLane `lightning.qubit` via adjoint differentiation. | **Verified** |
| **Quantum Encodings** | `qml/pitqsvdd/encodings/` | `AmplitudeEncoding`, `DataReuploadingEncoding`, `IQPZZEncoding`, `FeatureGroupEncoding` | Implements state preparation: $L_2$-normalized state amplitudes, multi-layer interleaved rotations, pairwise ZZ entangling phase maps, and wire group partitions. | **Verified** |
| **Flagship PI-TQSVDD** | `qml/pitqsvdd/models/temporal_pitqsvdd.py` | `TemporalPITQSVDD` (L31–L148) | Evaluates latent quantum trajectory $Z_t$, static SVDD distance to frozen centre $c$, causal GRU next-step prediction $\hat{z}_{t+1}$, and joint anomaly score $s_t = 0.5 s_{\text{norm}} + 0.5 s_{\text{temp}}$. | **Verified** |
| **Latent Predictor** | `qml/pitqsvdd/models/temporal_predictor.py` | `LatentTemporalPredictor` (L9–L42) | GRU network (hidden dim 16) with Linear + Tanh head predicting next latent quantum observable vector from history. Deterministically seeded ($seed=42$). | **Verified** |
| **Operational Event Metrics** | `qml/pitqsvdd/evaluation/event_metrics.py` | `compute_event_level_metrics` (L9–L79) | Groups predictions by `scenario_id`; tracks attack episodes; computes incident recall, detection delay from onset, and false alarms per uncompromised window. | **Verified** |
| **Persistence & Tables** | `scripts/run_complete_benchmark.py` | `QISMBenchmarkRunner.save_all_tables` (L959–L1065) | Serializes atomic checkpoints, `model_inventory.csv`, `per_run_metrics.csv`, `aggregate_metrics.csv`, `per_attack_metrics.csv`, and ablation tables. | **Verified** |

---

## 3. Dataset Integrity and Experimental Partitions

### 3.1 Observed Counts and Reconciliation

Direct inspection of `data/qism_research_dataset/` via PyArrow reveals the physical reality of the dataset:

| Partition | Scenarios | Simulation Runs | Raw Windows | Raw Normal | Raw Attack | Temporal Windows ($L=3$) | Temporal Normal | Temporal Attack |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`calibration`** | 20 | 40 | 800 | 800 | 0 | 760 | 760 | 0 |
| **`train`** | 60 | 180 | 3,600 | 3,600 | 0 | 3,480 | 3,480 | 0 |
| **`validation`** | 25 | 50 | 1,000 | 826 | 174 | 950 | 776 | 174 |
| **`test_in_distribution`** | 45 | 135 | 2,700 | 1,866 | 834 | 2,610 | 1,776 | 834 |
| **`test_generalization`** | 25 | 50 | 1,000 | 688 | 312 | 950 | 638 | 312 |
| **`test_stress`** | 25 | 50 | 1,000 | 740 | 260 | 950 | 690 | 260 |
| **Total** | **200** | **505** | **10,100** | **8,520** | **1,580** | **9,700** | **8,120** | **1,580** |

#### Reconciliation Explanation:
- In `QISM_EXPERIMENT_PROTOCOL.md`, counts are reported at the **raw physical window level** (2,700 in-distribution, 1,000 generalization, 1,000 stress).
- In `QISM_BENCHMARK_RESULTS.md`, counts are reported at the **temporal sequence level** (2,610 in-distribution, 950 generalization, 950 stress).
- Code audit of `_build_temporal_windows` (L336) proves that causal sequence construction requires $L=3$ windows. Because sequences are strictly constructed within scenarios, the first $L-1=2$ windows of each scenario cannot form a complete 3-step history and are dropped.
- Dropped window arithmetic:
  - `train`: $3,600 - (60 \times 2) = 3,480$ windows.
  - `validation`: $1,000 - (25 \times 2) = 950$ windows.
  - `test_in_distribution`: $2,700 - (45 \times 2) = 2,610$ windows.
  - `test_generalization`: $1,000 - (25 \times 2) = 950$ windows.
  - `test_stress`: $1,000 - (25 \times 2) = 950$ windows.
- Verification confirms that in the QISM simulation configuration, **attacks never begin at window index 0 or 1** (minimum attack onset is window 2). Consequently, exactly 0 attack windows are lost during temporal windowing, preserving 100% of attack ground-truth labels (834, 312, 260).

### 3.2 Attack Families and Distinct Incidents
Across `test_in_distribution`, 87 distinct attack episodes (834 attack windows) are represented across 6 attack families:
1. `intercept_resend`: 237 windows across 25 episodes.
2. `pns` (Photon-Number Splitting): 189 windows across 20 episodes.
3. `source_manipulation`: 129 windows across 14 episodes.
4. `channel_loss`: 111 windows across 12 episodes.
5. `detector_manipulation`: 108 windows across 11 episodes.
6. `timing` (Timing Delay): 60 windows across 5 episodes.

In `test_generalization`, 32 distinct attack episodes (312 attack windows) are evaluated.  
In `test_stress`, 28 distinct attack episodes (260 attack windows) are evaluated.

### 3.3 Partition Independence and Scenario Isolation
- **Scenario Independence:** Cross-partition set intersection confirms **zero scenario overlap** across all partitions. Scenarios are strictly disjoint.
- **Run Independence:** All 505 physical runs are generated under independent random number seeds derived from `master_seed = 20261019`.
- **Temporal Contamination:** The `_build_temporal_windows` method resets sliding sequences at scenario boundaries, preventing cross-scenario or cross-run leakage.

### 3.4 Leakage Controls and Excluded Attributes
- The allowlist `get_observable_feature_names()` yields exactly 76 features.
- Set intersection with `GROUND_TRUTH_WINDOW_COLUMNS` and `IDENTIFIER_WINDOW_COLUMNS` is empty (`set()`).
- Metadata fields (`scenario_id`, `run_id`, `window_id`, `ground_truth_attack_active`, `ground_truth_attack_type`, `ground_truth_attack_phase`, `ground_truth_attack_fraction`, `ground_truth_mean_severity`, `ground_truth_attack_id`) are excluded from model inputs and stored in separate metadata containers.

### 3.5 Physical Observability of Feature Set
- The 76 features represent observable physical quantities: detector counts, pulse statistics, timing histograms, SPAD dead-time dynamics, and sensor telemetry.
- **Subtlety Identified:** `estimated_channel_loss_db`, `channel_transmittance_variance`, and `channel_transmittance_drift` are computed from `events["channel_transmittance"]`. In actual QKD hardware, transmittance must be estimated via calibrated optical power meters or OTDR taps. In the simulator, this represents power-meter tap telemetry rather than an unobservable ground-truth parameter.
- `mean_channel_transmittance` is classified as simulator ground truth and is excluded from the 76 features.

### 3.6 Imputation of All-NaN Decoy Residuals
- Three decoy residual columns (`signal_decoy_residual`, `decoy_decoy_residual`, `vacuum_residual`) are all-NaN in `train` because decoy-state yield analysis was unconfigured during generation.
- Code inspection (L220–L227): `run_complete_benchmark.py` identifies `all_nan_cols`, fills them with 0.0 in `train`, and fits `SimpleImputer(strategy="median", fill_value=0.0)`. During test evaluation, test columns are filled with 0.0 before median imputation, preventing runtime failure or NaN propagation.

### 3.7 Preprocessing Isolation
- Scalers (`StandardScaler`, `RobustScaler`, `MinMaxScaler`), median imputers, and PCA projection matrices are fitted strictly on `train` (normal data) and applied to `validation` and `test_*` partitions without refitting.

### 3.8 Provenance Verification
- Manifest file `data/qism_research_dataset/dataset_manifest.json` verifies:
  - Configuration Hash: `80189033257fdeedf018b3afac859e9ffe422b2fc5019b05c899674d5eeb8499`
  - Simulator Version: `0.6.0`
  - Master Seed: `20261019`
  - Total Jobs: 505 completed, 0 failed.

---

## 4. Supervised versus Unsupervised Training Audit

The training data sources and label access across all evaluated model families are summarized below:

| Model / Family | Training Partition(s) | Training Samples ($N$) | Attack Labels During Fitting? | Validation Labels During Fitting? | Threshold Calibration Data | Test Data / Label Leakage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Statistical Baselines** (`Mahalanobis_Full`, `RobustCovariance_PCA6`) | `train` | 3,600 | **NO** | **NO** | `val` normal ($y=0$) | **NONE** |
| **Classical Unsupervised** (`IsolationForest_Full`, `IsolationForest_PCA6`, `OneClassSVM_PCA6`, `LocalOutlierFactor_PCA6`) | `train` | 3,600 | **NO** | **NO** | `val` normal ($y=0$) | **NONE** |
| **Classical Neural** (`ClassicalAutoencoder`, `ClassicalDeepSVDD`) | `train` | 3,600 | **NO** | **NO** | `val` normal ($y=0$) | **NONE** |
| **Classical Temporal** (`ClassicalTemporalAutoencoder`) | `train` | 3,480 ($L=3$) | **NO** | **NO** | `val` normal ($y=0$) | **NONE** |
| **Quantum Unsupervised** (`QuantumAutoencoder_4Q`, `StaticQuantumSVDD_4Q`) | `train` | 400 / 300 | **NO** | **NO** | `val` normal ($y=0$) | **NONE** |
| **Flagship PI-TQSVDD** (All 13 completed ablations & seeds) | `train` | 150 ($L=3$) | **NO** | **NO** | `val` normal ($y=0$) | **NONE** |
| **Supervised Classical** (`ClassicalLogisticRegression`, `RandomForestClassifier`, `SmallClassicalMLP`, `ClassicalRBFSVM`) | `train` (500 norm) + `validation` (174 atk) | 674 | **YES** | **YES (attack samples used for training)** | `val` normal ($y=0$) | **NONE** |
| **Supervised Quantum** (`VQC_4Q`) | `train` (400 norm) + `validation` (174 atk) | 574 | **YES** | **YES (attack samples used for training)** | `val` normal ($y=0$) | **NONE** |
| **Supervised Quantum Kernel** (`QuantumKernelSVM_4Q`) | `train` (60 norm) + `validation` (30 atk) | 90 | **YES** | **YES (attack samples used for training)** | `val` normal ($y=0$) | **NONE** |

### 4.1 Detailed Training Protocol Analysis

#### Unsupervised and One-Class Models
All one-class and unsupervised anomaly detectors are trained strictly on normal telemetry from `train`. Attack labels are never accessed during feature extraction, fitting, or optimization. The decision threshold $\tau$ is calibrated using the normal subset of validation scores ($y=0$) at target FPRs ($1\%$ and $5\%$). Test partitions are evaluated once with frozen thresholds.

#### Supervised Baselines and Validation Compromise
Lines 673–684 of `scripts/run_complete_benchmark.py`:
```python
X_train_norm = self.features["train"]["standard"][:500]
y_train_norm = np.zeros(500, dtype=int)

val_atk_mask = (self.metadata["validation"]["ground_truth_attack_active"] == True).values
X_val_atk = self.features["validation"]["standard"][val_atk_mask]
y_val_atk = np.ones(len(X_val_atk), dtype=int)

X_sup_train = np.vstack([X_train_norm, X_val_atk])
y_sup_train = np.concatenate([y_train_norm, y_val_atk])
```
- **Audit Finding:** The supervised baseline protocol uses all 174 attack windows from `validation` as positive training examples.
- **Consequence for Validation:** The `validation` partition contains 1,000 windows (826 normal, 174 attack). The supervised models are trained on the 174 attack windows from `validation`. Later, during threshold calibration (L408), the runner computes:
  ```python
  val_normal_scores = val_scores[val_y == 0]
  th_1pct = find_threshold_at_fpr(val_normal_scores, target_fpr=0.01)
  ```
- **Assessment:** While threshold calibration uses only normal validation scores ($y=0$), the model being evaluated was trained on the attack samples from that exact same partition. There is no separate, independent labeled development split.
- **Test Integrity:** The test partitions (`test_in_distribution`, `test_generalization`, `test_stress`) were never exposed during supervised fitting. The test evaluation is genuine, but the supervised baselines had access to the attack distribution from the validation partition.

---

## 5. Preprocessing and Feature Representation

### 5.1 Pipeline Flow
Telemetry features follow this transformation pipeline:
1. **Raw Vector:** $x_t \in \mathbb{R}^{76}$ observable features.
2. **All-NaN Infill:** `signal_decoy_residual`, `decoy_decoy_residual`, `vacuum_residual` infilled with 0.0.
3. **Median Imputation:** `SimpleImputer(strategy="median", fill_value=0.0)` fitted on `train`.
4. **Standardization:** `StandardScaler()` fitted on `train` ($\mu_{\text{train}}, \sigma_{\text{train}}$).
5. **Robust Scaling:** `RobustScaler()` fitted on `train`.
6. **PCA Projections:** `PCA(n_components=k, random_state=42)` fitted on standardized `train` for $k \in \{2, 4, 6, 8, 16\}$.
   - Variance explained: PCA-2 = 35.63%, PCA-4 = 46.49%, PCA-6 = 54.57%, PCA-8 = 60.07%, PCA-16 = 75.43%.
7. **Quantum Angle Scaling:** `MinMaxScaler(feature_range=(0.0, np.pi))` fitted on each PCA output.

### 5.2 Quantum Angle Scaling and Out-of-Range Handling
- In `scripts/run_complete_benchmark.py` (L252), `MinMaxScaler` is instantiated without `clip=True`.
- Consequently, test samples with PCA projections beyond the training min/max produce angles $< 0.0$ or $> \pi$.
- In PennyLane quantum circuits, $R_Y(\theta)$ gates handle arbitrary real values via $2\pi$ periodicity. However, angles $> \pi$ wrap around the Bloch sphere, which alters the topological mapping for extreme anomalies.

### 5.3 Quantum Encodings and Input Representation

| Model / Configuration | Feature Input | Dimension | State Preparation Circuit | Scale / Magnitude Preserved? |
| :--- | :--- | :---: | :--- | :---: |
| **`PITQSVDD_2Q_Angle`** | `pca_2_angle` | 2 | $R_Y(x_0) \otimes R_Y(x_1)$ | Yes ($[0, \pi]$ relative) |
| **`PITQSVDD_4Q_Angle_Ref`** | `pca_4_angle` | 4 | $\bigotimes_{j=0}^3 R_Y(x_j)$ | Yes ($[0, \pi]$ relative) |
| **`PITQSVDD_6Q_Angle`** | `pca_6_angle` | 6 | $\bigotimes_{j=0}^5 R_Y(x_j)$ | Yes ($[0, \pi]$ relative) |
| **`PITQSVDD_8Q_Angle`** | `pca_8_angle` | 8 | $\bigotimes_{j=0}^7 R_Y(x_j)$ | Yes ($[0, \pi]$ relative) |
| **`PITQSVDD_4Q_Amplitude`** | `pca_16_angle` | 16 | $\sum_{j=0}^{15} \frac{x_j}{\|x\|_2} \|j\rangle$ via `AmplitudeEmbedding` | **NO (Normalized to $\|\|\psi\|\|_2 = 1$)** |
| **`PITQSVDD_4Q_DataReuploading`** | `pca_16_angle` | 16 | Layer 0: $x_{0..3}$; Layer 1: $x_{4..7}$ via $R_Y$ | **Partial (Features 8..15 omitted)** |
| **`PITQSVDD_8Q_DataReuploading`** | `pca_16_angle` | 16 | Layer 0: $x_{0..7}$; Layer 1: $x_{8..15}$ via $R_Y$ | Yes (All 16 features injected) |
| **`PITQSVDD_4Q_IQPZZ`** | `pca_4_angle` | 4 | $H^{\otimes 4} \to R_Z(2x_j) \to \text{CNOT-RZ}(2(\pi-x_i)(\pi-x_j))\text{-CNOT}$ | Yes |
| **`PITQSVDD_4Q_FeatureGroup`** | `pca_4_angle` | 4 | Wire $j \gets R_Y(x_j)$ followed by ring CNOTs | Yes |

#### Critical Preprocessing Audit Findings:
1. **Amplitude Encoding Magnitude Loss:** `AmplitudeEncoding` normalizes the input feature vector by its Euclidean norm ($\|x\|_2$). This discards the absolute magnitude of telemetry features. Any attack that causes a uniform proportional scaling of features is mapped to the same quantum state.
2. **Data Reuploading Feature Truncation on 4 Qubits:** For `PITQSVDD_4Q_DataReuploading`, `inputs` has dimension 16 from PCA-16. With $n_{\text{wires}}=4$ and $n_{\text{layers}}=2$, `DataReuploadingEncoding.apply` calculates `start_idx = (layer_idx * 4) % 16`. Layer 0 receives features $0..3$ and Layer 1 receives features $4..7$. Features $8..15$ are never injected into the 4-qubit circuit. For 8 qubits (`PITQSVDD_8Q_DataReuploading`), Layer 0 receives $0..7$ and Layer 1 receives $8..15$, utilizing all 16 features.
3. **Feature Group Encoding Implementation:** `FeatureGroupEncoding` is documented as partitioning physical telemetry families into dedicated qubit registers. However, in `run_complete_benchmark.py` (L842), it receives `windows_pca4` (4 PCA features), assigning 1 PCA feature per wire with ring CNOTs. It does not operate on raw physical sensor groups.

---

## 6. PI-TQSVDD Mathematical and Implementation Audit

### 6.1 Quantum Representation
1. **State Preparation:** Encodes classical normalized input $x_t \in \mathbb{R}^d$ into state $|\psi_0(x_t)\rangle = U_{\text{enc}}(x_t)|0\rangle^{\otimes n}$.
2. **Variational Unitary:** Applies $M=2$ layers of parameterized rotations and periodic CNOT entanglement:
   $$W_l(\theta_l) = U_{\text{cnot}} \prod_{j=0}^{n-1} R_Z(\theta_{l, j, 2}) R_Y(\theta_{l, j, 1}) R_X(\theta_{l, j, 0})$$
3. **Observables Measured:** Pauli-$Z$ expectation values on all wires:
   $$z_{\tau, j} = \langle \psi(x_\tau, \theta) | Z_j | \psi(x_\tau, \theta) \rangle \in [-1, 1], \quad j \in \{0, \dots, n-1\}$$
4. **Hardware Parameter Accounting:**
   - 2 Qubits: 4 CNOTs, 12 single-qubit rotations, 12 trainable parameters.
   - 4 Qubits: 8 CNOTs, 24 single-qubit rotations, 24 trainable parameters.
   - 6 Qubits: 12 CNOTs, 36 single-qubit rotations, 36 trainable parameters.
   - 8 Qubits: 16 CNOTs, 48 single-qubit rotations, 48 trainable parameters.
5. **Backend & Differentiation:** PennyLane `lightning.qubit` statevector simulator with exact adjoint backpropagation (`diff_method="adjoint"`).
6. **Parameter Initialization:** Quantum weights are initialized randomly away from zero: $\theta \sim \mathcal{N}(0.1, 0.2^2)$.

### 6.2 One-Class SVDD Objective and Anti-Collapse Safeguards
- **Centre Initialization:** In `TemporalPITQSVDD.init_centre` (L57–L70), the centre $c \in \mathbb{R}^n$ is computed as the empirical mean of the first 150 normal training sequences:
  $$c_j = \frac{1}{N_{\text{init}}} \sum_{i=1}^{N_{\text{init}}} z_{i, t, j}$$
- **Perturbation:** Coordinates with $|c_j| < 10^{-4}$ are perturbed to $0.1$ to prevent zero-gradient representation collapse.
- **Freezing:** $c$ is registered as a PyTorch buffer (`self.register_buffer("centre", ...)`), not an `nn.Parameter`. It is frozen throughout training.
- **SVDD Loss:**
  $$\mathcal{L}_{\text{SVDD}} = \frac{1}{N} \sum_{i=1}^N \|z_{i, t} - c\|_2^2$$

### 6.3 Temporal Latent Dynamics Objective
- **Architecture:** `LatentTemporalPredictor` consists of a 1-layer GRU (input dim $n$, hidden dim 16) with a Linear + Tanh projection head.
- **Target:** Predicts the **next latent quantum state** $z_{t+1} \in \mathbb{R}^n$, NOT the raw telemetry vector.
- **Target Source:** True next latent state is obtained by passing the next PCA vector $x_{t+1}$ through the quantum circuit: $z_{t+1} = \text{circuit}(x_{t+1})$.
- **Masking:** If $t+1$ crosses a scenario boundary, `has_target = False` and the sequence is excluded from prediction loss.
- **Temporal Loss:**
  $$\mathcal{L}_{\text{temporal}} = \frac{1}{N_{\text{valid}}} \sum_{i \in \text{valid}} \|z_{i, t+1} - \hat{z}_{i, t+1}\|_2^2$$

### 6.4 Total Objective Function
The implemented loss function is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SVDD}} + \lambda_t \mathcal{L}_{\text{temporal}} + \lambda_{\text{reg}} \left( \|\theta\|_2^2 + \|\psi\|_2^2 \right)$$
where $\lambda_t = 0.5$ (or $0.0$ in ablation), and $\lambda_{\text{reg}} = 10^{-4}$.

### 6.5 Training Duration Reality
- In `scripts/run_complete_benchmark.py` (L895–L908), PI-TQSVDD models are trained on **only the first 150 sequences of `train` for exactly 8 gradient iterations** using Adam ($\text{lr}=0.02$).
- Training on the full 3,480 temporal sequences was omitted due to CPU quantum simulation bottlenecks.

---

## 7. Physics-Informed Assessment

### 7.1 Meaning of "Physics-Informed" in the Implemented Code
A critical question for research integrity is whether PI-TQSVDD implements an analytical physics loss or is simply an anomaly detector operating on physical telemetry.

Audit of `qml/pitqsvdd/models/temporal_pitqsvdd.py` confirms:
1. **No Physics Loss Terms:** The loss contains only hypersphere distance, GRU temporal prediction error, and weight decay. There are no decoy-state yield bounds, transmittance constraints, or Hamiltonian energy penalties in the objective.
2. **Where Physical Knowledge Enters:**
   - **Feature Derivation:** The 76 input features are derived from physical models of single-photon detectors, optical pulse statistics, and fiber channels.
   - **Temporal Regularization:** The causal temporal loss enforces continuity over 1 ms physical sampling intervals, penalizing unphysical abrupt jumps during uncompromised operation.
   - **Decoy-State Invariants:** While decoy residual features were intended to provide physical invariants, in the actual dataset they are all-NaN in `train` and imputed to 0.0.

### 7.2 Distinction for Publication
The model is a **Physics-Telemetry-Informed Temporal Quantum Anomaly Detector**. It does not enforce explicit physical constraints through penalty terms in the objective function.

---

## 8. Scoring, Thresholds, and Operational Metrics

### 8.1 Scoring Functions and Directionality
Across all 30 models, larger scores indicate more anomalous behavior:
- Classical Distance / Reconstruction: Euclidean norm $\|x - \hat{x}\|_2^2$ or Mahalanobis distance.
- Decision Function Baselines: Isolation Forest and One-Class SVM return $-f(x)$.
- Supervised Baselines: Predicted probability of attack $P(y=1|x)$.
- PI-TQSVDD:
  $$s_t = 0.5 \cdot \|z_t - c\|_2^2 + 0.5 \cdot \|z_t - \hat{z}_t\|_2^2$$
  where $\hat{z}_t = g_\psi(Z_{t-1})$.

### 8.2 Threshold Calibration Procedure
1. Validation normal scores are isolated: $S_{\text{val\_norm}} = \{s(x) \mid x \in \mathcal{D}_{\text{val}}, y=0\}$.
2. Thresholds are calibrated via empirical percentiles:
   $$\tau_{1\%} = \text{Percentile}(S_{\text{val\_norm}}, 99.0)$$
   $$\tau_{5\%} = \text{Percentile}(S_{\text{val\_norm}}, 95.0)$$
3. Calibrated thresholds are frozen and applied to test partitions.

### 8.3 Statistical Limitations of 1% FPR Calibration
- Validation normal sample count is $N_{\text{val\_norm}} = 826$ for point models and $776$ for temporal models.
- At $N=776$, each sample corresponds to $1 / 776 \approx 0.129\%$ of the distribution. The 99th percentile isolates the top 7–8 samples.
- **Audit Finding:** The 1% FPR is an empirical calibration target on validation, not an exact bound on test. Actual test FPR ranges from $0.05\%$ to $4.33\%$ depending on distribution shifts.

### 8.4 Event-Level Operational Metrics
Implemented in `qml/pitqsvdd/evaluation/event_metrics.py`:
- **Incident Recall:** An attack episode is counted as detected if at least one alarm is raised between attack onset and offset:
  $$\text{Incident Recall} = \frac{\text{Detected Episodes}}{\text{Total Episodes}}$$
- **Detection Delay:** Measured as elapsed windows from episode start to first alarm:
  $$\Delta t = t_{\text{first\_alarm}} - t_{\text{onset}}$$
  If detected on the first window of injection, delay is 0.
- **False Alarm Rate (FAR):** False alarms divided by total benign windows during uncompromised operation.

---

## 9. Statistical Validity Audit

### 9.1 Multi-Seed Stability Sweep
Four seeds ($42, 101, 2024, 777$) were evaluated for `PITQSVDD_4Q_Angle_Ref`:

| Seed | In-Dist AUROC | In-Dist AUPRC | Recall @ 1% FPR | Incident Recall | Generalization AUROC | Stress AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42 (Ref)** | 0.6751 | 0.5634 | 5.5% | 27.6% | 0.6952 | 0.5322 |
| **101** | 0.7066 | 0.6099 | 9.2% | 32.2% | 0.6990 | 0.5634 |
| **2024** | 0.6945 | 0.5819 | 4.6% | 24.1% | 0.7032 | 0.5531 |
| **777** | 0.6801 | 0.5771 | 10.4% | 34.5% | 0.6911 | 0.5317 |
| **Mean $\pm$ Std** | **$0.6891 \pm 0.0143$** | **$0.5831 \pm 0.0197$** | **$7.4\% \pm 2.8\%$** | **$29.6\% \pm 4.6\%$** | **$0.6971 \pm 0.0052$** | **$0.5451 \pm 0.0153$** |

- **Verification:** The reported mean ($0.6891$) and standard deviation ($0.0142$ / $0.0143$) in `PI_TQSVDD_ABLATION_REPORT.md` match the recorded seed results in `per_run_metrics.csv`.
- **Implementation Nuance:** The multi-seed sweep varies the seed passed to `VariationalQuantumCircuit`, altering quantum parameter initialization. However, `LatentTemporalPredictor` has a hardcoded internal `seed=42`, meaning the classical predictor weights are initialized identically across all multi-seed runs.

### 9.2 Audit of the Claimed $p < 0.01$ Significance Result
- **Claim in Documentation:** Question 19 of `QISM_BENCHMARK_RESULTS.md` states:
  *"The improvement of temporal modeling over static modeling ($\Delta \text{AUROC} = +2.98\% - +11.63\%$) is statistically credible ($p < 0.01$ across scenario-level bootstrap resamples)."*
- **Audit Result (UNSUPPORTED):** Comprehensive search of `scripts/run_complete_benchmark.py`, `qml/`, and `tests/` found **no code implementing bootstrap resampling, paired t-tests, Wilcoxon signed-rank tests, or p-value calculations** for model comparisons.
- **Recommendation:** This claim must be removed or properly computed using an independent scenario-level paired bootstrap script before publication.

---

## 10. Benchmark Artifact Reconciliation

### 10.1 Model Count Reconciliation
- `model_inventory.csv`: Lists 28 models.
- `docs/QISM_REPRODUCIBILITY.md`: States "Trains all 29 models".
- `results/qism_model_benchmark/tables/aggregate_metrics.csv`: Contains exactly 30 evaluated models.
- **Reconciliation:**
  - 31 runs were scheduled in `run_complete_benchmark.py`: 9 classical, 4 supervised, 4 quantum baselines, and 14 PI-TQSVDD configurations.
  - `PITQSVDD_4Q_ContextAware` failed during execution (`AttributeError: 'ContextAwarePITQSVDD' object has no attribute 'init_centre'`) and was recorded in `failed_runs_table.csv`.
  - Exactly 30 models completed evaluation ($31 - 1 = 30$).
  - `model_inventory.csv` hardcoded 28 planned models (omitting seeds 101, 2024, 777, and incorrectly listing ContextAware as completed).

### 10.2 Discrepancy Register

| Target Entity | Documented / Table Claim | Observed Reality in Code / Artifacts | Status | Root Cause & Required Correction |
| :--- | :--- | :--- | :--- | :--- |
| **In-Distribution Test Windows** | 2,700 windows (`PROTOCOL`) vs 2,610 (`BENCHMARK`) | 2,700 raw windows; 2,610 length-3 temporal sequences | **Discrepancy (Reconciled)** | Sliding windows drop 2 warm-up windows per scenario ($45 \times 2 = 90$). Update documentation to state both raw and sequence counts. |
| **Generalization & Stress Windows** | 1,000 windows (`PROTOCOL`) vs 950 (`BENCHMARK`) | 1,000 raw windows; 950 length-3 temporal sequences | **Discrepancy (Reconciled)** | Sliding windows drop 2 warm-up windows per scenario ($25 \times 2 = 50$). Clarify in documentation. |
| **Evaluated Model Count** | 28 (`INVENTORY`), 29 (`REPRODUCIBILITY`), 30 (`AGGREGATE`) | 30 completed models; 1 failed run | **Discrepancy** | `PITQSVDD_4Q_ContextAware` crashed. `model_inventory.csv` hardcoded 28 without seed runs. Synchronize all tables to 30 evaluated models. |
| **Context-Aware PI-TQSVDD** | Marked COMPLETED in `model_inventory.csv` | Recorded in `failed_runs_table.csv` (`AttributeError`) | **Discrepancy** | Runner called `.init_centre()` instead of `.init_centres()`. Fix method name in runner and update inventory status to FAILED. |
| **Supervised Training Split** | Documented as "independent development split" | Uses validation attack windows ($N=174$) for training | **Discrepancy** | Supervised baselines consume validation attack data. Document that validation is not an independent split for supervised baselines. |
| **Quantum Kernel SVM Scale** | Ranked #2 on master leaderboard ($0.7570$) | Evaluated on only 150 windows with 2 of 6 attack types | **Discrepancy** | Evaluated with `max_samples=150` due to Gram matrix cost. Must be annotated on leaderboard with an asterisk denoting reduced sample evaluation. |
| **Statistical Significance ($p<0.01$)** | Reported in Question 19 of `BENCHMARK_RESULTS.md` | No statistical hypothesis testing code exists | **Unverified** | Claim is unsupported by codebase. Remove p-value claim or execute paired bootstrap script. |
| **Decoy-State Residuals** | Included in 76-feature allowlist | All-NaN in `train`; imputed to 0.0 | **Verified with Caveat** | Decoy state yield calculation was unconfigured. Features contribute constant 0.0 after imputation. |
| **Data Reuploading on 4 Qubits** | 16 PCA features injected across layers | Only first 8 features injected across 2 layers | **Discrepancy** | Slicing logic leaves features 8..15 un-injected for $n_q=4, M=2$. Accurately document in ablation report. |
| **Amplitude Encoding Scale** | Encodes 16 features | Normalizes vector to unit sphere, discarding scale | **Verified with Caveat** | Euclidean norm is discarded. Document that absolute magnitude information is eliminated. |

---

## 11. Scientific Conclusions Supported by the Evidence

Based strictly on the audited source code, dataset, and verified artifact tables:

1. **Strongest Overall In-Distribution Detection:**
   **`ClassicalTemporalAutoencoder`** (AUROC $0.7616$, AUPRC $0.6921$). It demonstrates the highest discrimination across the full 2,610 test sequences.
2. **Strongest Out-of-Distribution Generalization:**
   **`ClassicalTemporalAutoencoder`** (Gen AUROC $0.8355$), followed by `Mahalanobis_Full` ($0.7858$), `PITQSVDD_4Q_Amplitude` ($0.7761$), and `ClassicalAutoencoder` ($0.7758$).
3. **Best Recall at Calibrated 1% FPR:**
   **`ClassicalLogisticRegression`** ($40.29\%$), followed by `Mahalanobis_Full` ($40.05\%$), `ClassicalAutoencoder` ($39.81\%$), and **`PITQSVDD_4Q_Amplitude`** ($38.25\%$).
4. **Best Incident-Level Recall and Detection Latency:**
   - Incident Recall: `IsolationForest_PCA6` ($67.82\%$), `Mahalanobis_Full` ($67.82\%$), `VQC_4Q` ($67.82\%$), `PITQSVDD_4Q_DataReuploading` ($66.67\%$), `ClassicalLogisticRegression` ($65.52\%$), and `PITQSVDD_4Q_Amplitude` ($64.37\%$).
   - Latency: `ClassicalRBFSVM` ($0.32$ windows), `ClassicalLogisticRegression` ($0.98$ windows), and `PITQSVDD_4Q_Amplitude` ($1.05$ windows).
5. **Top Quantum Model Comparison:**
   `PITQSVDD_4Q_Amplitude` is the strongest fully evaluated quantum model (ID AUROC $0.7147$, Gen AUROC $0.7761$, Stress AUROC $0.6390$, Latency $1.05$ windows). It matches classical dense autoencoders ($0.7127$), but is outperformed by the classical temporal autoencoder ($0.7616$).
6. **Benefit of Temporal Modeling in PI-TQSVDD:**
   Ablation confirms that temporal latent modeling improves performance: `PITQSVDD_4Q_Angle_Ref` ($\lambda_t=0.5$, AUROC $0.6751$) outperforms `PITQSVDD_4Q_NoTemporal` ($\lambda_t=0.0$, AUROC $0.6453$) by $+2.98\%$ AUROC and $+3.65\%$ AUPRC. It also outperforms baseline `StaticQuantumSVDD_4Q` ($0.6402$) by $+3.49\%$ AUROC and improves incident recall from $21.84\%$ to $27.59\%$.
7. **Optimal Qubit and Encoding Configuration:**
   4 qubits with Amplitude Encoding (`PITQSVDD_4Q_Amplitude`) provides the best trade-off, achieving $0.7147$ AUROC. For angle encoding, 4 qubits ($0.6751$) outperforms 2 qubits ($0.5909$, information bottleneck) and 8 qubits ($0.5908$, state dispersion). Data reuploading rescues 8-qubit performance to $0.6941$.
8. **Failure Modes Under Stress:**
   IQP/ZZ phase encoding fails ($0.4838$ AUROC, below chance) because artificial pairwise ZZ interactions over-correlate disparate sensor streams. Quantum Kernel SVM degrades severely under stress ($0.2980$).
9. **Supported Quantum Representation Conclusions:**
   Quantum circuits can learn compact support vector representations on physical telemetry, and amplitude encoding provides an effective compression scheme matching classical feedforward autoencoders.
10. **Unsupported Quantum Advantage Claims:**
   **No claim of quantum computational advantage, quantum supremacy, or hardware readiness is supported.** Classical temporal models achieve higher detection accuracy ($0.7616$ vs $0.7147$), train $100\times$ faster ($0.38\text{ s}$ vs $33.38\text{ s}$), and perform inference $2,000\times$ to $8,000\times$ faster ($3.7\,\mu\text{s}$ vs $8\text{--}14\text{ ms}$). Furthermore, simulated circuits omit physical quantum hardware noise (decoherence, gate infidelities, SPAM errors).

---

## 12. Limitations and Recommended Corrections

### High Priority (Pre-Publication Corrections)
1. **Annotate Quantum Kernel SVM Sample Size:** The master leaderboard and text must explicitly disclose that `QuantumKernelSVM_4Q` was evaluated on a reduced subset ($N=150$, covering only 2 attack types) due to Gram matrix scaling, and that its stress AUROC collapsed to $0.2980$.
2. **Clarify Supervised Baseline Training Split:** Document that supervised models were trained using attack windows from the validation partition, meaning validation is not an independent evaluation split for those models.
3. **Remove or Calculate Statistical Significance:** Remove the claim of $p < 0.01$ bootstrap significance from `QISM_BENCHMARK_RESULTS.md` Question 19, or implement and execute a dedicated paired bootstrap test across scenario clusters.
4. **Clarify "Physics-Informed" Terminology:** Explicitly clarify in the technical report that physical domain knowledge enters through engineered telemetry features and temporal sequence continuity, not through analytical physical constraint terms in the loss function.
5. **Harmonize Window Counts:** Explicitly explain in `QISM_EXPERIMENT_PROTOCOL.md` and `QISM_BENCHMARK_RESULTS.md` that 2,700/1,000/1,000 are raw telemetry windows and 2,610/950/950 are length-3 causal sequences resulting from dropping 2 warm-up windows per scenario.

### Medium Priority (Code Quality & Robustness)
6. **Fix Context-Aware PI-TQSVDD Method Call:** In `scripts/run_complete_benchmark.py` (L899), change `model.init_centre(t_seqs)` to check `hasattr(model, "init_centres")`, allowing `PITQSVDD_4Q_ContextAware` to execute without raising an `AttributeError`.
7. **Synchronize Model Inventory Table:** Update `results/qism_model_benchmark/tables/model_inventory.csv` to include all 30 evaluated models and mark `PITQSVDD_4Q_ContextAware` as FAILED.
8. **Pass Multi-Seed Parameter to Temporal Predictor:** In `TemporalPITQSVDD.__init__`, pass `seed=config.seed` to `LatentTemporalPredictor` rather than using the hardcoded default `seed=42`.
9. **Decoy Residual Feature Implementation:** Either implement the complete decoy-state yield analysis in the simulator to generate non-zero decoy residual values, or remove the all-NaN columns from the observable feature allowlist.

---

## 13. Reproducibility Appendix

The following non-destructive commands and verification checks were executed during this audit:

### Environment and Hardware Accounting
```powershell
# Python runtime check
.venv\Scripts\python.exe -c "import torch, pennylane as qml, pyarrow; print('PyTorch:', torch.__version__, '| PennyLane:', qml.__version__, '| Device: lightning.qubit verified!')"
```

### Partition Count and Temporal Arithmetic Verification
```powershell
.venv\Scripts\python.exe -c "
import pandas as pd
from pathlib import Path

for sp in ['train', 'validation', 'test_in_distribution', 'test_generalization', 'test_stress']:
    p = Path(f'data/qism_research_dataset/{sp}')
    dfs = [pd.read_parquet(f) for f in sorted(p.rglob('telemetry.parquet'))]
    df = pd.concat(dfs, ignore_index=True)
    n_scen = df['scenario_id'].nunique()
    n_runs = df['run_id'].nunique()
    total_w = len(df)
    norm_w = (df['ground_truth_attack_active'] == False).sum()
    atk_w = (df['ground_truth_attack_active'] == True).sum()
    temp_w = total_w - n_scen * 2
    temp_norm = norm_w - n_scen * 2
    print(f'{sp:22s} | Scenarios: {n_scen:3d} | Runs: {n_runs:3d} | Raw Total: {total_w:4d} (Norm: {norm_w:4d}, Atk: {atk_w:4d}) | Temp Total: {temp_w:4d} (Norm: {temp_norm:4d}, Atk: {atk_w:4d})')
"
```

### PCA Explained Variance Verification
```powershell
.venv\Scripts\python.exe -c "
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from qism.telemetry.features import get_observable_feature_names

cols = get_observable_feature_names()
dfs = [pd.read_parquet(f) for f in sorted(Path('data/qism_research_dataset/train').rglob('telemetry.parquet'))]
df = pd.concat(dfs, ignore_index=True)
df[[c for c in cols if df[c].isna().all()]] = 0.0
X_imp = SimpleImputer(strategy='median', fill_value=0.0).fit_transform(df[cols])
X_std = StandardScaler().fit_transform(X_imp)

for k in [2, 4, 6, 8, 16]:
    pca = PCA(n_components=k, random_state=42).fit(X_std)
    print(f'PCA-{k:2d}: {np.sum(pca.explained_variance_ratio_)*100:.2f}% variance')
"
```

### Multi-Seed AUROC Mean and Standard Deviation Verification
```powershell
.venv\Scripts\python.exe -c "
import pandas as pd, numpy as np
df = pd.read_csv('results/qism_model_benchmark/tables/per_run_metrics.csv')
seeds = ['PITQSVDD_4Q_Angle_Ref', 'PITQSVDD_4Q_Seed101', 'PITQSVDD_4Q_Seed2024', 'PITQSVDD_4Q_Seed777']
sub = df[(df['model_id'].isin(seeds)) & (df['partition'] == 'test_in_distribution')]
aurocs = sub['auroc'].values
print('Mean AUROC:', np.mean(aurocs))
print('Sample Std (ddof=1):', np.std(aurocs, ddof=1))
"
```

---
*Report certified by Senior Research Software Auditor.*
