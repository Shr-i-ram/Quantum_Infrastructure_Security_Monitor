# Quantum Infrastructure Security Monitor (QISM)
## Complete Model and Baseline Inventory

This document provides a comprehensive, rigorous inventory of every model, baseline, and anomaly detector implemented and evaluated in the QISM research benchmark.

---

## 1. Executive Summary

The QISM benchmark systematically evaluates **29 models and configurations** across four major families:
1. **Classical Statistical and Non-Neural Baselines**: Empirical and robust distance metrics, tree-based isolation, density-based local outlier detection.
2. **Classical Neural and Temporal Models**: Symmetric multilayer autoencoders, recurrent GRU sequence autoencoders, and one-class hypersphere representation learning (Deep SVDD).
3. **Classical Supervised Classifiers**: Regularized logistic regression, Random Forest ensembles, Support Vector Classifiers with RBF kernels, and classical MLPs.
4. **Quantum Machine Learning (QML) Baselines**: Variational Quantum Classifiers (VQC), Quantum Kernel SVMs (QSVM), and unsupervised Quantum Autoencoders (QAE).
5. **Flagship PI-TQSVDD and Ablations**: Physics-Informed Temporal Quantum Support Vector Data Description across register sizes ($n \in \{2, 4, 6, 8\}$ qubits), diverse encoding strategies (Angle, Data Reuploading, Amplitude, IQP/ZZ, Feature Group, Learned Bottleneck), and objective ablations (static vs. causal temporal vs. context-aware).

---

## 2. Comprehensive Model Registry

| Model Identifier | Family | Learning Paradigm | Input Representation | Parameter Count / Complexity | Training Assumption | Checkpoint Support | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`IsolationForest_Full`** | Classical Statistical | Unsupervised Anomaly Detection | 76 Observable Features | 100 Trees ($7,600$ splits) | Normal Training Data Only | Joblib (`.joblib`) | COMPLETED |
| **`IsolationForest_PCA6`** | Classical Statistical | Unsupervised Anomaly Detection | PCA-6 ($54.57\%$ variance) | 100 Trees ($600$ splits) | Normal Training Data Only | Joblib (`.joblib`) | COMPLETED |
| **`OneClassSVM_PCA6`** | Classical Statistical | Unsupervised Novelty Detection | PCA-6 ($54.57\%$ variance) | Support Vectors ($\nu = 0.05$) | Normal Training Data Only | Joblib (`.joblib`) | COMPLETED |
| **`LocalOutlierFactor_PCA6`** | Classical Statistical | Unsupervised Density Novelty | PCA-6 ($54.57\%$ variance) | $k=20$ Neighbors | Normal Training Data Only | Joblib (`.joblib`) | COMPLETED |
| **`Mahalanobis_Full`** | Classical Statistical | Parametric Distance Metric | 76 Observable Features | Mean $\mu \in \mathbb{R}^{76}$, $\Sigma^{-1} \in \mathbb{R}^{76 \times 76}$ | Normal Training Data Only | NPZ (`.npz`) | COMPLETED |
| **`RobustCovariance_PCA6`** | Classical Statistical | Robust Minimum Covariance Det | PCA-6 ($54.57\%$ variance) | MCD Mean & Covariance ($6 \times 6$) | Normal Training Data Only | Joblib (`.joblib`) | COMPLETED |
| **`ClassicalAutoencoder`** | Classical Neural | Reconstruction Anomaly Detection | 76 Observable Features | Dense MLP ($76 \to 32 \to 8 \to 32 \to 76$) | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`ClassicalTemporalAutoencoder`** | Classical Temporal | Sequence Reconstruction | 76 Features ($L=3$ sliding window) | GRU Encoder-Decoder ($24$ hidden) | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`ClassicalDeepSVDD`** | Classical Neural | Hypersphere Normality Learning | 76 Observable Features | Bias-free MLP ($76 \to 24 \to 4$) | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`ClassicalLogisticRegression`** | Classical Supervised | Binary Classification | 76 Observable Features | Linear Weights $w \in \mathbb{R}^{76}$, Bias $b \in \mathbb{R}$ | Labeled Development Split | Joblib (`.joblib`) | COMPLETED |
| **`RandomForestClassifier`** | Classical Supervised | Ensembled Decision Trees | 76 Observable Features | 100 Trees (Max Depth 12) | Labeled Development Split | Joblib (`.joblib`) | COMPLETED |
| **`SmallClassicalMLP`** | Classical Supervised | Compact Neural Classification | PCA-4 ($46.49\%$ variance) | $4 \to 8 \to 1$ ($49$ parameters) | Labeled Development Split | PyTorch (`.pt`) | COMPLETED |
| **`ClassicalRBFSVM`** | Classical Supervised | Maximum Margin Classification | PCA-4 ($46.49\%$ variance) | Support Vectors ($C=1.0$) | Labeled Development Split | Joblib (`.joblib`) | COMPLETED |
| **`VQC_4Q`** | Quantum Supervised | Variational Quantum Classification | PCA-4 Angle Embedding ($4$ qubits) | $2$ layers ($24$ quantum, $5$ classical) | Labeled Development Split | PyTorch (`.pt`) | COMPLETED |
| **`QuantumKernelSVM_4Q`** | Quantum Kernel | Quantum State Fidelity Kernel | PCA-4 Angle + CNOT Ring ($4$ qubits) | Precomputed Fidelity Gram Matrix | Labeled Development Split | Joblib (`.joblib`) | COMPLETED |
| **`QuantumAutoencoder_4Q`** | Quantum Unsupervised | Trash Subsystem Excitation | PCA-4 Angle Embedding ($4$ qubits) | $2$ layers ($2$ latent, $2$ trash wires) | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`StaticQuantumSVDD_4Q`** | Quantum One-Class | Quantum State Hypersphere SVDD | PCA-4 Angle Embedding ($4$ qubits) | $2$ layers ($24$ quantum parameters) | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`PITQSVDD_2Q_Angle`** | Flagship PI-TQSVDD | Temporal Quantum SVDD | PCA-2 Angle Embedding ($2$ qubits, $L=3$) | $2$ layers + GRU Predictor | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`PITQSVDD_4Q_Angle_Ref`** | Flagship PI-TQSVDD | Temporal Quantum SVDD (Reference) | PCA-4 Angle Embedding ($4$ qubits, $L=3$) | $2$ layers + GRU Predictor | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`PITQSVDD_6Q_Angle`** | Flagship PI-TQSVDD | Temporal Quantum SVDD | PCA-6 Angle Embedding ($6$ qubits, $L=3$) | $2$ layers + GRU Predictor | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`PITQSVDD_8Q_Angle`** | Flagship PI-TQSVDD | Temporal Quantum SVDD | PCA-8 Angle Embedding ($8$ qubits, $L=3$) | $2$ layers + GRU Predictor | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`PITQSVDD_4Q_DataReuploading`** | Flagship PI-TQSVDD | Temporal Reuploading Quantum SVDD | PCA-16 Interleaved ($4$ qubits, $2$ blocks) | $2$ blocks ($24$ parameters) + GRU | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`PITQSVDD_8Q_DataReuploading`** | Flagship PI-TQSVDD | Temporal Reuploading Quantum SVDD | PCA-16 Interleaved ($8$ qubits, $2$ blocks) | $2$ blocks ($48$ parameters) + GRU | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`PITQSVDD_4Q_Amplitude`** | Flagship PI-TQSVDD | State Vector Amplitude SVDD | 16 Normalized Features ($2^4$ amplitudes) | $2$ layers ($24$ parameters) + GRU | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`PITQSVDD_4Q_IQPZZ`** | Flagship PI-TQSVDD | Entangling IQP/ZZ Feature Map | PCA-4 Angle + ZZ Entanglement ($4$ qubits) | $2$ layers + GRU Predictor | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`PITQSVDD_4Q_FeatureGroup`** | Flagship PI-TQSVDD | Domain Feature Group SVDD | Partitioned Feature Registers ($4$ qubits) | $2$ layers + GRU Predictor | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`PITQSVDD_4Q_NoTemporal`** | Flagship PI-TQSVDD | Static SVDD Ablation ($\lambda_t = 0.0$) | PCA-4 Angle Embedding ($4$ qubits, $L=3$) | $2$ layers ($24$ parameters) | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |
| **`PITQSVDD_4Q_ContextAware`** | Flagship PI-TQSVDD | Multi-Regime Context SVDD | PCA-4 Angle Embedding ($4$ qubits, $L=3$) | Multi-centre Hyperspheres | Normal Training Data Only | PyTorch (`.pt`) | COMPLETED |

---

## 3. Mathematical Objectives by Family

### 3.1 Unsupervised Hypersphere SVDD (Classical & Quantum)
Minimizes the squared Euclidean distance of latent representations to reference centre vector $c$:
$$\mathcal{L}_{\mathrm{SVDD}} = \frac{1}{N} \sum_{i=1}^{N} \|\phi(x_i) - c\|_2^2 + \lambda \|\theta\|_2^2$$
where:
- $\phi(x) = f_{\mathrm{neural}}(x)$ for Classical Deep SVDD.
- $\phi(x) = \langle \psi(x, \theta) | Z | \psi(x, \theta) \rangle$ for Quantum SVDD.
- Reference centre $c$ is initialized strictly from normal training representations away from zero ($c \approx \mathbb{E}_{x \sim \mathcal{D}_{\mathrm{train}}}[\phi(x)]$, $\epsilon = 0.1$) to prevent representation collapse.

### 3.2 Flagship PI-TQSVDD Combined Loss
Combines static quantum hypersphere normality learning with causal temporal latent prediction:
$$\mathcal{L}_{\mathrm{PI-TQSVDD}} = \mathcal{L}_{\mathrm{SVDD}} + \lambda_t \mathcal{L}_{\mathrm{temporal}} + \lambda_{\mathrm{reg}} \mathcal{R}(\theta, \psi)$$
where:
$$\mathcal{L}_{\mathrm{temporal}} = \frac{1}{N_{\mathrm{valid}}} \sum_{i \in \mathrm{valid}} \|z_{i, t+1} - \hat{z}_{i, t+1}\|_2^2$$
$$\hat{z}_{i, t+1} = g_\psi([z_{i, t-L+1}, \dots, z_{i, t}])$$
Continuous anomaly scoring combines static and dynamical deviation:
$$s_t = \alpha \|z_t - c\|_2^2 + \beta \|z_t - \hat{z}_t\|_2^2$$

### 3.3 Quantum Autoencoder (QAE) Compression
Maps 4 qubits into 2 latent qubits ($\mathcal{H}_A$) and 2 trash qubits ($\mathcal{H}_B$). The unsupervised objective penalizes excitations in the trash subsystem:
$$\mathcal{L}_{\mathrm{QAE}} = 1.0 - \frac{1}{N} \sum_{i=1}^{N} P_i(\mathrm{trash} = |00\rangle)$$
$$s(x) = 1.0 - P(\mathrm{trash} = |00\rangle) \in [0, 1]$$

### 3.4 Quantum Kernel Support Vector Machine (QSVM)
Computes state-overlap fidelity Gram matrix via PennyLane:
$$K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2 = |\langle 0000 | U^\dagger(x_j) U(x_i) | 0000 \rangle|^2$$
Dual optimization maximizes:
$$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \sum_i \alpha_i y_i = 0$$

---

## 4. Feature Space Governance and Leakage Safeguards

1. **Permitted Observable Feature Space**: Exactly 76 observable physical telemetry features are permitted as model inputs.
2. **Strict Exclusions**:
   - `ground_truth_attack_active` (Binary label)
   - `ground_truth_attack_type` (Categorical attack family)
   - `ground_truth_attack_phase` (Attack lifecycle phase)
   - `ground_truth_attack_fraction` (Intensity fraction)
   - `ground_truth_mean_severity` (Severity score)
   - `ground_truth_attack_id` (Unique attack incident ID)
   - `scenario_id`, `run_id`, `window_id` (Identifiers)
3. **Partition Isolation**: Preprocessing parameters (medians, means, standard deviations, PCA projection matrices, angle bounds) are fitted exclusively on the `train` partition (3,600 normal windows) and stored as frozen artifacts.
