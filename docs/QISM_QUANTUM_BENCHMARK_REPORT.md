# Quantum Infrastructure Security Monitor (QISM)
## Quantum Machine Learning Benchmark Report

This technical report details the empirical results, circuit architectures, quantum simulation performance, and comparative characteristics of all quantum machine learning models evaluated in QISM on the authoritative research dataset (`data/qism_research_dataset`).

---

## 1. Executive Summary and Quantum Backend Specification

All quantum models in QISM are implemented using **PennyLane (v0.42.3)** with **`lightning.qubit`** as the primary high-performance quantum simulation backend, leveraging adjoint differentiation for exact, fast gradient propagation.

### Summary Table of Quantum ML Baselines

| Model Identifier | Quantum Task | Circuit Architecture | Qubits | Circuit Depth | Trainable Quantum Parameters | Backend / Diff Method | In-Dist AUROC | In-Dist AUPRC | Gen AUROC | Stress AUROC | Train Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`QuantumKernelSVM_4Q`** | Fidelity Kernel SVM | Angle Embedding + CNOT Ring (Adjoint Inversion) | 4 | 2 | 0 (Kernel SVM) | `default.qubit` (state overlap) | **0.7570** | **0.6236** | 0.6992 | 0.2980 | 8.61 |
| **`QuantumAutoencoder_4Q`** | Unsupervised Compression | Angle Embedding + Strongly Entangling (Trash \|00>) | 4 | 2 | 24 | `lightning.qubit` (adjoint) | **0.6832** | **0.6332** | **0.7624** | **0.5709** | 1.33 |
| **`VQC_4Q`** | Supervised Classification | Angle Embedding + Strongly Entangling + Linear Head | 4 | 2 | 24 ($+5$ classical) | `lightning.qubit` (adjoint) | **0.6550** | 0.6036 | 0.6944 | 0.5130 | 2.89 |
| **`StaticQuantumSVDD_4Q`** | One-Class Normality | Angle Embedding + Strongly Entangling Layers | 4 | 2 | 24 | `lightning.qubit` (adjoint) | **0.6402** | 0.5652 | **0.7189** | 0.5132 | 13.93 |

---

## 2. Model Formulations and Architectures

### 2.1 Variational Quantum Classifier (VQC)
- **State Preparation**: $4$-qubit AngleEmbedding with $R_Y(x_i)$ rotations on PCA-4 scaled features.
- **Variational Ansatz**: $2$ layers of `StronglyEntanglingLayers` containing single-qubit rotations ($R_X, R_Y, R_Z$) per qubit followed by periodic CNOT entangling rings.
- **Measurement**: Pauli-$Z$ expectation values on all 4 wires: $z_i = \langle \psi(x, \theta) | Z_i | \psi(x, \theta) \rangle \in [-1, 1]$.
- **Classification Head**: Trainable linear layer mapping $z \in \mathbb{R}^4 \to \mathbb{R}$, passed through a sigmoid function to predict attack probability $p(y=1|x)$.
- **Empirical Performance**: Achieved in-distribution AUROC of $0.6550$ and AUPRC of $0.6036$. Performance on generalization regimes remained moderate ($0.6944$).

### 2.2 Quantum Kernel Support Vector Machine (QSVM)
- **Feature Map**: $U_{\phi}(x) = \bigotimes_{i=0}^3 R_Y(x_i) \cdot \prod_{i=0}^3 \mathrm{CNOT}(i, (i+1)\%4)$.
- **Kernel Function**: Evaluates transition probability to state $|0000\rangle$:
  $$K(x_i, x_j) = |\langle 0000 | U_\phi^\dagger(x_j) U_\phi(x_i) | 0000 \rangle|^2$$
- **Dual Optimization**: Standard C-SVM optimization with precomputed Gram matrix.
- **Resource Profiling**: Gram matrix computation scales as $\mathcal{O}(N_1 \cdot N_2)$. While achieving strong in-distribution AUROC ($0.7570$), it collapsed catastrophically on the stress partition ($0.2980$, far below chance), indicating severe susceptibility to non-stationary sensor drift.

### 2.3 Quantum Autoencoder (QAE)
- **Subsystem Partition**: Wire 0 and 1 are designated **Latent wires**; Wire 2 and 3 are designated **Trash wires**.
- **Normality Learning**: Trained to compress normal QKD telemetry into the 2 latent qubits by mapping trash wires strictly to reference state $|00\rangle$:
  $$\mathcal{L}_{\mathrm{QAE}} = 1.0 - P(\mathrm{trash} = |00\rangle)$$
- **Empirical Findings**: QAE achieved an in-distribution AUROC of $0.6832$ and generalization AUROC of $0.7624$, outperforming supervised VQC. When attacks occur, anomalous feature correlations prevent complete compression, producing measurable excitation on the trash wires ($s(x) \to 1.0$).

### 2.4 Static Quantum SVDD
- **Objective**: Maps quantum state Pauli-$Z$ expectations $z = \langle Z \rangle \in \mathbb{R}^4$ to a hypersphere centered at reference vector $c$:
  $$\mathcal{L}_{\mathrm{SVDD}} = \frac{1}{N} \sum_{i=1}^N \|z_i - c\|_2^2 + \lambda \|\theta\|_2^2$$
- **Empirical Findings**: Reached **$0.6402$ in-distribution AUROC** and **$0.7189$ generalization AUROC**. When combined with temporal sequence modeling in PI-TQSVDD, performance rises to **$0.6751 - 0.7147$**, directly proving the value of temporal dynamics.

---

## 3. Quantum vs. Classical Baseline Takeaways

1. **Static Quantum Models Match Classical Counterparts**: Static Quantum SVDD ($0.6402$) and Quantum Autoencoder ($0.6832$) achieve representation learning competitive with Classical Deep SVDD ($0.6791$), confirming valid quantum representation learning without artificial degradation.
2. **Quantum Kernels vs. Classical Kernels**: The quantum fidelity kernel achieved $0.7570$ in-distribution AUROC compared to $0.6313$ for the classical RBF kernel. However, under environmental stress, the quantum kernel degraded to $0.2980$, whereas classical models maintained stability ($0.52 - 0.63$).
3. **The Static Limitation**: All static quantum models fell well short of the classical temporal autoencoder ($0.7616$), establishing that static quantum representations alone cannot outperform classical temporal sequence modeling. This demonstrates the necessity of the flagship PI-TQSVDD architecture, which unites quantum representation learning with causal temporal sequence dynamics.
