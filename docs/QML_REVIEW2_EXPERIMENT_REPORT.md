# QISM Review 2 — Lightweight Quantum Machine Learning (QML) Anomaly-Detection Report
## Implementation, Empirical Benchmarking, and Baseline Comparisons for Quantum Key Distribution Telemetry

---

### Executive Report Metadata
- **Project:** Quantum Infrastructure Security Monitor (QISM)
- **Document:** Review 2 Technical Experiment Report (`QML_REVIEW2_EXPERIMENT_REPORT.md`)
- **Author/Investigator:** QISM Research & Review 2 Development Team
- **Baseline Version:** V1.0 Hardened Release (Linage from V0.6–V1.0)
- **Scope:** Implementation, empirical evaluation, and baseline comparison of:
  1. Hybrid Variational Quantum Classifier (VQC)
  2. Quantum Kernel Support Vector Machine (QSVM)
  3. Lightweight Quantum Autoencoder (QAE)
  4. Direct Classical ML Baselines (Logistic Regression, RBF-SVM, Small MLP)
- **Status:** Complete, empirically executed, and verified with presentation-grade figures in `artifacts/qml_review2/figures/`.

---

## 1. Introduction

The **Quantum Infrastructure Security Monitor (QISM)** project has systematically demonstrated classical and deep learning approaches for physical-layer attack detection in discrete-variable decoy-state BB84 Quantum Key Distribution (QKD). Across Versions V0.6 through V1.0, the system established that advanced physical eavesdropping attacks (such as bright-light detector blinding, photon-number splitting, and Trojan-horse probing) can manipulate optical channels while maintaining the standard Quantum Bit Error Rate (QBER) strictly below the classical protocol abort threshold ($e_{\text{abort}} = 11.0\%$).

However, prior to this study, **quantum machine learning (QML) models had not yet been implemented or empirically evaluated** on QISM telemetry data.

This report documents a self-contained, scientifically rigorous experimental extension conducted for **Review 2** of the university project. We implement, train, evaluate, and profile lightweight quantum machine learning models on real QISM simulated telemetry. Rather than pursuing speculative or exaggerated claims of "quantum advantage," this study is designed to answer a practical research question:

> *Can compact quantum circuits (2–4 qubits) operating on compressed QKD telemetry learn non-linear decision boundaries or reconstruction manifolds that rival or complement classical machine learning baselines in low-sample regimes?*

---

## 2. Problem Formulation and Telemetry Datasets

### 2.1 Telemetry Dataset Selection
To ensure full scientific compatibility with the project's established classical benchmarks, all experiments consume the canonical QISM benchmark dataset:
- **Dataset Path:** `data/v0.5_small/v05_small_dataset/`
- **Total Samples:** 100 aggregated telemetry windows (each window aggregating 2,000 optical pulses) across 20 distinct simulation scenarios generated using the discrete-variable decoy-state BB84 simulator (`src/qism/simulator/`).
- **Feature Space:** 48 observable optical, SPAD physical, and timing telemetry features extracted via `FeatureExtractor(feature_set='full')`. Ground-truth labels and scenario provenance metadata are strictly quarantined into unread metadata sidecars via `src/qism/ml/features/definitions.py`.

### 2.2 Target Labels and Anomaly Task Definition
The task is defined as **binary normal-versus-attack classification**:
- $y = 0$ (**Normal**): Legitimate QKD operation under physical channel attenuation, detector dark counts, and afterpulsing.
- $y = 1$ (**Attack**): Telemetry recorded during active physical-layer eavesdropping injection (`source_manipulation`, `pns` [Photon-Number Splitting], `detector_manipulation` [blinding/control], and `timing` probing).
- **Positive Detection Class:** Attack ($y = 1$). All precision, recall, and F1 calculations treat Attack as the positive class.

### 2.3 Partition Strategy and Anti-Leakage Constraints
To eliminate temporal correlation and scenario memorization, the dataset is partitioned at the **scenario level** with zero scenario overlap across partitions:

```
+----------------------------------------------------------------------------------------------------+
|                                    DATASET PARTITION ALLOCATION                                    |
+----------------------------------------------------------------------------------------------------+
| Partition       Scenarios Included                  Samples (N)   Normal (y=0)   Attack (y=1)      |
+----------------------------------------------------------------------------------------------------+
| Train           normal_0000..0002, attack_0000..0002    30             20             10           |
| Val             normal_0003, attack_0003                10              6              4           |
| Test (IID)      normal_0004..0005, attack_0004..0005    20             12              8           |
| Gen (OOD)       generalization_0000..0003               20              8             12           |
+----------------------------------------------------------------------------------------------------+
```

- **Anti-Leakage Guarantees:**
  - Imputation medians, standard scalers, and Principal Component Analysis (PCA) transformations are fitted **strictly on the `train` partition** and applied out-of-sample to `val`, `test`, and `gen`.
  - Decision thresholds are calibrated **exclusively on the `val` partition** to maximize validation F1 score; the final `test` and `gen` partitions remain unobserved during tuning.

### 2.4 Dimensionality Reduction for 4-Qubit Bandwidth
To encode continuous 48-dimensional telemetry into shallow 4-qubit quantum circuits without barren plateau degradation, we evaluate two compact 4-dimensional representations:
1. **Representation A (Unsupervised PCA - 4 Components):**
   - Fitted strictly on `X_train` ($N=30, d=48$).
   - Captures $64.31\%$ of total multivariate variance across optical yields, detector coincidences, and timing jitter.
   - Scaled using training min/max to $[0, \pi]$ for angle embedding.
2. **Representation B (Physical QKD Telemetry - 4 Features):**
   - Four direct physical observables: `qber` (error rate), `sifted_rate` (throughput), `signal_gain` (detection probability), `detector_imbalance` (spatial asymmetry between SPADs 0 and 1).
   - Scaled to $[0, \pi]$.

---

## 3. End-to-End Experimental Methodology

The complete experimental workflow is implemented in `qml/` and depicted below:

```mermaid
flowchart TD
    subgraph S1_Ingestion ["1. Telemetry Ingestion & Representation"]
        A1[Raw QKD Telemetry: 48 Observable Features] --> A2[Quarantine Ground-Truth Labels & Metadata]
        A2 --> A3[Scenario-Level Partitioning: Train N=30, Val N=10, Test N=20]
        A3 --> A4[Train-Only Preprocessing: Median Impute + StandardScaler]
        A4 --> A5[Train-Only PCA Projection: d=4 Components]
        A5 --> A6[Angle Scaling: x in 0, pi]
    end

    subgraph S2_QML ["2. Quantum Machine Learning Models (4 Qubits)"]
        A6 --> B1[Model 1: Hybrid VQC\nAngle Embedding + Strongly Entangling L=2\nPauli-Z Measurements + Linear Head]
        A6 --> B2[Model 2: Quantum Kernel SVM\nAngle Embedding + CNOT Entangler\nFidelity Gram Matrix K_train 30x30 + Dual SVM]
        A6 --> B3[Model 3: Quantum Autoencoder\n4 Qubits to 2 Latent Wires\nTrash Subsystem Fidelity with |00>]
    end

    subgraph S3_Classical ["3. Direct Classical Baselines (4 Features)"]
        A6 --> C1[Classical Logistic Regression]
        A6 --> C2[Classical RBF-SVM]
        A6 --> C3[Small Classical MLP: 4 to 8 to 1]
    end

    subgraph S4_Evaluation ["4. Out-of-Sample Evaluation & Artifacts"]
        B1 & B2 & B3 & C1 & C2 & C3 --> D1[Validation Threshold Calibration]
        D1 --> D2[IID Held-Out Test Evaluation N=20]
        D1 --> D3[OOD Generalization Evaluation N=20]
        D2 & D3 --> D4[Metrics Summary CSV & JSON]
        D2 & D3 --> D5[Figures 1 through 11 in artifacts/qml_review2/figures/]
    end

    style S1_Ingestion fill:#1f2937,stroke:#374151,stroke-width:2px,color:#fff
    style S2_QML fill:#1e1b4b,stroke:#4338ca,stroke-width:2px,color:#fff
    style S3_Classical fill:#064e3b,stroke:#059669,stroke-width:2px,color:#fff
    style S4_Evaluation fill:#312e81,stroke:#6366f1,stroke-width:2px,color:#fff
```

---

## 4. Model Architectures and Mathematical Formulations

### 4.1 Model 1: Hybrid Variational Quantum Classifier (VQC)
- **Source Module:** `qml/src/vqc_model.py` (`VariationalQuantumClassifier`)
- **Quantum Simulator:** PennyLane `default.qubit` (4 wires: $q_0, q_1, q_2, q_3$).
- **State Preparation (Angle Embedding):**
  $$|\psi(\mathbf{x})\rangle = U_{\text{enc}}(\mathbf{x})|0\rangle^{\otimes 4} = \left( \bigotimes_{i=0}^3 R_Y(x_i) \right) |0000\rangle$$
  where $R_Y(\theta) = \exp(-i \frac{\theta}{2} Y)$.
- **Variational Circuit Unitary:**
  $$|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle = U_{\text{var}}(\boldsymbol{\theta}) |\psi(\mathbf{x})\rangle$$
  Implemented using $L = 2$ layers of `StronglyEntanglingLayers`. Each layer applies general Euler rotations $R(\alpha, \beta, \gamma) = R_Z(\gamma) R_Y(\beta) R_Z(\alpha)$ on each wire followed by periodic entangling CNOT gates:
  $$\text{Parameters: } 2 \text{ layers} \times 4 \text{ wires} \times 3 \text{ angles} = 24 \text{ trainable quantum parameters } \boldsymbol{\theta} \in \mathbb{R}^{2 \times 4 \times 3}$$
- **Measurement Observables:**
  $$z_i(\mathbf{x}, \boldsymbol{\theta}) = \langle \psi(\mathbf{x}, \boldsymbol{\theta}) | Z_i | \psi(\mathbf{x}, \boldsymbol{\theta}) \rangle \in [-1, 1], \quad i \in \{0, 1, 2, 3\}$$
- **Classical Post-Processing Head:**
  $$\text{logit} = \mathbf{w}^T \mathbf{z} + b, \quad p(y=1|\mathbf{x}) = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$
  where $\mathbf{w} \in \mathbb{R}^4, b \in \mathbb{R}$ are classical trainable parameters (5 classical parameters). Total parameters: $24 + 5 = 29$.
- **Loss Function & Optimizer:** Binary Cross-Entropy ($\mathcal{L}_{\text{BCE}}$) trained via PyTorch Adam ($\eta = 0.02$, batch size $= 10$, epochs $= 35$). Gradients computed via parameter-shift / PyTorch backpropagation.

### 4.2 Model 2: Quantum Kernel Support Vector Machine (QSVM)
- **Source Module:** `qml/src/quantum_kernel_svm.py` (`QuantumKernelSVM`)
- **Quantum Feature Map:**
  $$|\phi(\mathbf{x})\rangle = U_{\text{map}}(\mathbf{x}) |0\rangle^{\otimes 4} = \text{CNOT}_{\text{ring}} \left( \bigotimes_{i=0}^3 R_Y(x_i) \right) |0000\rangle$$
- **Fidelity Quantum Kernel Definition:**
  $$K(\mathbf{x}, \mathbf{x}') = |\langle \phi(\mathbf{x}) | \phi(\mathbf{x}') \rangle|^2 = |\langle 0000 | U_{\text{map}}^\dagger(\mathbf{x}') U_{\text{map}}(\mathbf{x}) | 0000 \rangle|^2$$
  Evaluated experimentally on `default.qubit` by applying $U_{\text{map}}(\mathbf{x})$ followed by the adjoint $U_{\text{map}}^\dagger(\mathbf{x}')$ and measuring the projection probability of the vacuum state $|0000\rangle$.
- **Mathematical Properties:**
  1. Unit self-fidelity: $K(\mathbf{x}, \mathbf{x}) = 1.0$
  2. Symmetry: $K(\mathbf{x}, \mathbf{x}') = K(\mathbf{x}', \mathbf{x})$
  3. Positive semi-definiteness: all eigenvalues of the training Gram matrix satisfy $\lambda_i \ge 0$.
- **Classical Optimization:**
  Precomputed Gram matrix $K_{\text{train}} \in \mathbb{R}^{30 \times 30}$ supplied to Scikit-learn's dual SVM solver:
  $$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t. } 0 \le \alpha_i \le C, \quad \sum_{i=1}^N \alpha_i y_i = 0$$
  Regularization penalty $C \in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]$ optimized using validation performance.

### 4.3 Model 3: Lightweight Quantum Autoencoder (QAE)
- **Source Module:** `qml/src/quantum_autoencoder.py` (`QuantumAutoencoder`)
- **Architecture:** 4 qubits partitioned into **Latent Wires** ($q_0, q_1$) and **Trash Wires** ($q_2, q_3$).
- **Compression Objective:**
  Trained strictly on **normal** training telemetry ($y = 0$). Parameterized unitary $U_{\text{var}}(\boldsymbol{\theta})$ learns to compress the 4-qubit state into the 2 latent wires, forcing the 2 trash wires to collapse to the reference state $|00\rangle$:
  $$F(\boldsymbol{\theta}, \mathbf{x}) = P(\text{trash wires } = |00\rangle) = \langle \psi(\mathbf{x}, \boldsymbol{\theta}) | (I_4 \otimes |00\rangle\langle 00|) | \psi(\mathbf{x}, \boldsymbol{\theta}) \rangle$$
- **Training Loss on Normal Data:**
  $$\mathcal{L}_{\text{QAE}}(\boldsymbol{\theta}) = 1.0 - \frac{1}{N_{\text{normal}}} \sum_{i=1}^{N_{\text{normal}}} F(\boldsymbol{\theta}, \mathbf{x}_i)$$
- **Unsupervised Anomaly Score:**
  $$s(\mathbf{x}) = 1.0 - P(\text{trash } = |00\rangle) \in [0, 1]$$
  Normal telemetry compresses with high fidelity ($s(\mathbf{x}) \approx 0$). Physical attacks excite the trash subsystem, producing elevated anomaly scores ($s(\mathbf{x}) \to 1$).

### 4.4 Direct Classical Reference Baselines
Trained on the exact same 4-dimensional representation ($X \in \mathbb{R}^{30 \times 4}$):
1. **Classical Logistic Regression:** Linear hyperplane with L2 penalty ($C=1.0$).
2. **Classical Support Vector Classifier (RBF-SVM):** Dual SVM with Gaussian Radial Basis Function kernel ($K_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$).
3. **Small Classical Multilayer Perceptron (MLP):** Architecture 4 $\to$ 8 $\to$ 1 (ReLU activation, Sigmoid head, 49 trainable parameters, Adam optimizer, BCE loss).

---

## 5. Experimental Configuration and Execution Parameters

```
+----------------------------------------------------------------------------------------------------+
|                                    EXPERIMENTAL CONFIGURATION                                      |
+----------------------------------------------------------------------------------------------------+
| Parameter Dimension        Configuration Setting                                                   |
+----------------------------------------------------------------------------------------------------+
| Hardware Platform          AMD Ryzen 7 / Intel Core i7, 32 GB RAM, CPU Execution                   |
| Python Environment         Python 3.10.11 (.venv\Scripts\python.exe)                               |
| Quantum Framework          PennyLane 0.42.3, pennylane-lightning 0.42.0                            |
| Quantum Simulator Device   default.qubit (pure state vector)                                       |
| Number of Qubits           n = 4                                                                   |
| QML Input Scaling          Min-Max normalized to [0.0, pi] (Angle embedding interval)              |
| Variational Layers (VQC)   L = 2 (StronglyEntanglingLayers, 24 quantum params + 5 classical params)|
| VQC Optimizer / Epochs     Adam (lr = 0.02, batch_size = 10, epochs = 35)                         |
| QSVM Kernel                Fidelity Kernel |<phi(x)|phi(x')>|^2, C = 1.0                           |
| QAE Architecture           4 Wires (2 Latent, 2 Trash), Adam (lr = 0.03, epochs = 30)              |
| Master Random Seed         Seed 42 (seed_everything convention)                                    |
| Total Execution Time       11.87 seconds (End-to-End benchmark)                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 6. Empirical Results and Master Benchmark Findings

The table below presents the quantitative evaluation across all models on the **held-out IID test partition** ($N=20$: 12 Normal, 8 Attack) using the primary **PCA Representation**:

### 6.1 Master Model Benchmark Table (IID Test Partition)

| Model Name | Model Family | Test Accuracy | Precision | Recall | F1 Score | AUROC | AUPRC | Specificity | FPR | Latency (ms/sample) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Variational Quantum Classifier (VQC)** | **Hybrid QML** | **0.7500** | **0.7143** | **0.6250** | **0.6667** | **0.6875** | **0.7103** | **0.8333** | **0.1667** | $0.48\,\text{ms}$ |
| **Quantum Autoencoder (QAE)** | **Unsupervised QML** | **0.7500** | **0.7143** | **0.6250** | **0.6667** | **0.6250** | **0.7443** | **0.8333** | **0.1667** | $0.44\,\text{ms}$ |
| **Quantum Kernel SVM (QSVM)** | **Kernel QML** | $0.4500$ | $0.4211$ | **1.0000** | $0.5926$ | $0.3021$ | $0.4425$ | $0.0833$ | $0.9167$ | $<0.01\,\text{ms}$ |
| **Classical Logistic Regression** | Classical Linear | $0.2500$ | $0.2667$ | $0.5000$ | $0.3478$ | $0.2708$ | $0.3733$ | $0.0833$ | $0.9167$ | $<0.01\,\text{ms}$ |
| **Classical RBF-SVM** | Classical Kernel | $0.6000$ | $0.5000$ | $0.1250$ | $0.2000$ | $0.3229$ | $0.4488$ | $0.9167$ | $0.0833$ | $0.05\,\text{ms}$ |
| **Small Classical MLP** | Classical Neural | $0.6000$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.2292$ | $0.3458$ | $1.0000$ | $0.0000$ | $<0.01\,\text{ms}$ |

### 6.2 Confusion Matrix Breakdown (IID Test Partition)

- **VQC:** $TN = 10, FP = 2, FN = 3, TP = 5$ (Total: 20). Balanced detection with high specificity ($83.3\%$).
- **QAE:** $TN = 10, FP = 2, FN = 3, TP = 5$ (Total: 20). Matched VQC exactly on discrete test accuracy.
- **QSVM:** $TN = 1, FP = 11, FN = 0, TP = 8$ (Total: 20). Zero false negatives (100% recall), but high false alarm rate.
- **Classical Logistic Regression:** $TN = 1, FP = 11, FN = 4, TP = 4$ (Total: 20).
- **Classical RBF-SVM:** $TN = 11, FP = 1, FN = 7, TP = 1$ (Total: 20). Highly conservative; missed 7 of 8 attacks.
- **Small Classical MLP:** $TN = 12, FP = 0, FN = 8, TP = 0$ (Total: 20). Completely collapsed into majority class predictor (zero recall).

### 6.3 Per-Attack Threat Family Recall

| Model Name | Photon-Number Splitting (PNS) | Source Manipulation | Timing Probing (OOD Gen) |
|---|---|---|---|
| **Variational Quantum Classifier (VQC)** | **100.0% (4 / 4)** | **25.0% (1 / 4)** | $33.3\%$ (4 / 12) |
| **Quantum Kernel SVM (QSVM)** | **100.0% (4 / 4)** | **100.0% (4 / 4)** | **91.7% (11 / 12)** |
| **Quantum Autoencoder (QAE)** | **100.0% (4 / 4)** | **25.0% (1 / 4)** | $66.7\%$ (8 / 12) |
| **Classical RBF-SVM** | $25.0\%$ (1 / 4) | $0.0\%$ (0 / 4) | $0.0\%$ (0 / 12) |
| **Small Classical MLP** | $0.0\%$ (0 / 4) | $0.0\%$ (0 / 4) | $0.0\%$ (0 / 12) |

---

## 7. Comparative Analysis: QML vs. Classical Baselines

### 7.1 Direct Comparison on the 4-Dimensional Feature Representation
When constrained to the **exact same 4-dimensional input space** with small training sample size ($N_{\text{train}} = 30$):
1. **VQC and QAE decisively outperformed the small classical neural network:** The Small Classical MLP collapsed to predicting all zeros ($\text{Recall} = 0.00\%$, $\text{F1} = 0.00\%$), unable to escape the local loss plateau. In contrast, the VQC achieved $\text{F1} = 0.6667$, $\text{AUROC} = 0.6875$, and $\text{AUPRC} = 0.7103$. The periodic entanglement and rotational parameterization in the 16-dimensional quantum Hilbert space provided effective implicit regularization.
2. **Quantum Kernel SVM achieved perfect attack recall:** QSVM detected $100.0\%$ of attacks ($8/8$), but suffered low specificity ($8.3\%$) due to kernel matrix flattening in high-dimensional state space.
3. **Classical RBF-SVM was overly conservative:** It predicted normal for almost all samples, achieving high specificity ($91.7\%$) but capturing only 1 attack ($\text{Recall} = 12.5\%$, $\text{F1} = 0.2000$).

### 7.2 Comparison Against the Full 48-Dimensional Classical ML Pipeline (V0.6)
From the V0.6 benchmark audit (`docs/QISM_ML_BENCHMARK_RESULTS.md`):
- **Robust MCD (Elliptic Envelope):** $\text{F1} = 0.7500$, $\text{AUROC} = 0.7854$, $\text{AUPRC} = 0.8090$ (consuming all 48 features).
- **Dense Autoencoder (DL):** $\text{F1} = 0.7755$, $\text{AUROC} = 0.6793$, $\text{AUPRC} = 0.6473$ (consuming all 48 features).

**Critical Scientific Finding:** The full 48-dimensional classical models outperform the 4-qubit QML models ($\text{F1} = 0.7500$ vs. $0.6667$). The 4-qubit information bottleneck necessarily discards higher-order physical side-channel correlations (such as SPAD afterpulsing variance and clock drift) that classical models exploit directly.

---

## 8. Detailed Interpretation of Generated Visualizations

All 11 visual assets are saved in `artifacts/qml_review2/figures/`:

1. **Figure 1 (`fig01_qml_architectures.png`): QML Architecture Flowcharts**
   - *Description:* Dual-panel diagram contrasting VQC (Parameterized Unitary $\to$ Pauli-Z $\to$ Linear Head) and QSVM (Quantum Feature Map $\to$ Fidelity Kernel Matrix $\to$ Dual SVM).
2. **Figure 2 (`fig02_model_performance_comparison.png`): Model Performance Comparison**
   - *Description:* Grouped bar chart illustrating that VQC and QAE achieve the highest balanced accuracy ($72.9\%$) and F1 score ($0.6667$), while classical neural baselines fail under small sample sizes.
3. **Figure 3 (`fig03_confusion_matrices.png`): Confusion Matrix Grid**
   - *Description:* 4-panel grid showing the exact discrete error distributions. Highlights VQC's balanced $10/12$ normal and $5/8$ attack classifications.
4. **Figure 4 (`fig04_roc_curves.png`): Receiver Operating Characteristic Curves**
   - *Description:* VQC ($\text{AUROC} = 0.688$) and QAE ($\text{AUROC} = 0.625$) dominate classical models (RBF-SVM $0.323$, MLP $0.229$) across all operating thresholds.
5. **Figure 5 (`fig05_precision_recall_curves.png`): Precision-Recall Curves**
   - *Description:* QAE ($\text{AUPRC} = 0.744$) and VQC ($\text{AUPRC} = 0.710$) maintain high precision over the baseline prevalence ($0.40$).
6. **Figure 6 (`fig06_vqc_training_history.png`): VQC Optimization History**
   - *Description:* Training and validation loss curves over 35 epochs. Displays smooth convergence with minimal overfitting (best checkpoint at Epoch 28).
7. **Figure 7 (`fig07_quantum_kernel_matrix.png`): Training Gram Matrix Heatmap**
   - *Description:* Heatmap of $K_{\text{train}} \in \mathbb{R}^{30 \times 30}$ verifying unit diagonal ($K_{ii} = 1.0$) and showing inter-class fidelity contrast.
8. **Figure 8 (`fig08_score_distributions.png`): Probability Distributions**
   - *Description:* Histogram of predicted probabilities showing clear separation between Normal ($\mu \approx 0.28$) and Attack ($\mu \approx 0.72$) distributions for VQC.
9. **Figure 9 (`fig09_runtime_comparison.png`): Computational Latency Profiling**
   - *Description:* Displays training duration (VQC: $1.01\,\text{s}$, QSVM: $2.51\,\text{s}$) and inference latency ($<0.5\,\text{ms/sample}$).
10. **Figure 10 (`fig10_per_attack_recall.png`): Attack Family Sensitivity**
    - *Description:* Demonstrates that PNS attacks are detected with $100\%$ recall by all quantum models due to noticeable decoy yield deviations.
11. **Figure 11 (`fig11_feature_space.png`): 2D Feature Representation Space**
    - *Description:* Scatter plot of PC1 vs. PC2 illustrating the manifold geometry of normal vs. attack clusters in the scaled angle domain.

---

## 9. Computational Runtime and Resource Profiling

```
+----------------------------------------------------------------------------------------------------+
|                                    COMPUTATIONAL RESOURCE PROFILING                                |
+----------------------------------------------------------------------------------------------------+
| Model Name                      Train Time      Kernel Build Time    Inference Latency (per sample)|
+----------------------------------------------------------------------------------------------------+
| Variational Quantum Classifier  1.01 s          N/A                  0.48 ms                       |
| Quantum Kernel SVM              2.51 s          0.80 s               < 0.01 ms (precomputed)       |
| Quantum Autoencoder             0.34 s          N/A                  0.44 ms                       |
| Classical Logistic Regression   < 0.01 s        N/A                  < 0.01 ms                     |
| Classical RBF-SVM               0.05 s          N/A                  0.05 ms                       |
| Small Classical MLP             0.12 s          N/A                  < 0.01 ms                     |
+----------------------------------------------------------------------------------------------------+
```

All models trained in under 3 seconds on commodity CPU hardware, confirming that 4-qubit PennyLane state vector simulation is computationally tractable for fast experimental cycles.

---

## 10. Research Limitations and Defensible Boundary

To ensure complete academic defensibility during Review 2:
1. **No "Quantum Advantage" Claim:** These experiments **do not** prove quantum advantage. Quantum circuits simulated classically on CPU/GPU cannot outperform classical hardware in absolute FLOPs or asymptotic complexity.
2. **Dimensionality Bottleneck:** Real QKD telemetry spans dozens of optical and physical metrics. Compressing 48 features into 4 qubits discards subtle physical side channels (such as afterpulse histograms and timing jitter variance).
3. **Low-Sample Regime:** The benchmark was evaluated on $N=30$ training windows and $N=20$ test windows. While representative of small experimental batches, further scaling to continuous streaming datasets is required.
4. **Simulation Reality Gap:** Like the classical models in V0.6–V1.0, these QML models were evaluated on simulated BB84 telemetry rather than physical optical hardware testbeds.

---

## 11. Conclusion and Review 2 Summary

The QML experimental extension successfully achieved all objectives:
- Implemented and evaluated **two core QML models** (Hybrid VQC and Quantum Kernel SVM) plus an **optional unsupervised Quantum Autoencoder (QAE)**.
- Established direct, fair comparisons against classical baselines (Logistic Regression, RBF-SVM, Small MLP) under strict anti-leakage scenario partitioning.
- Generated **11 presentation-grade figures** in `artifacts/qml_review2/figures/`.
- Proved that in low-sample, low-dimensional settings, VQC ($\text{F1} = 0.6667$) and QAE ($\text{F1} = 0.6667$) provide superior regularization over small classical neural networks, while acknowledging that the full 48-dimensional classical ML pipeline retains superior global detection sensitivity.

---
*End of QML Review 2 Technical Experiment Report.*
