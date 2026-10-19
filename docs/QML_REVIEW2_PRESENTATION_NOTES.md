# QISM Review 2 — Presentation Notes and Oral Defense Guide
## Quantum Machine Learning (QML) Anomaly-Detection Extension for QKD Infrastructure

---

### Document Overview
This guide provides concise, technically rigorous speaking notes and answers to likely review questions for **University Project Review 2**. All empirical claims, metrics, and comparisons are grounded directly in the experimental outputs produced during this study (`artifacts/qml_review2/`).

---

## Part A: The 60-Second Elevator Pitch

> *"In the QISM project, our core objective is detecting physical eavesdropping attacks on Quantum Key Distribution networks that operate below the standard 11% QBER protocol abort threshold. While our V0.6–V1.0 work demonstrated strong classical ML and deep learning baselines, quantum machine learning had not yet been explored.*
>
> *For Review 2, we implemented and evaluated three lightweight QML models using PennyLane on 4 qubits: a Hybrid Variational Quantum Classifier (VQC), a Quantum Kernel Support Vector Machine (QSVM), and an unsupervised Quantum Autoencoder (QAE), evaluated on the canonical QISM telemetry benchmark under strict zero-leakage scenario partitioning.*
>
> *Our empirical findings show that in a low-sample, 4-dimensional feature regime, the VQC and Quantum Autoencoder achieved an F1 score of 0.667 and AUROC of 0.688, outperforming small classical neural networks which collapsed to zero recall. However, when compared against our full 48-feature classical pipeline, classical models like Robust MCD still achieve superior detection (F1 = 0.750) because current 4-qubit quantum bandwidth forces an information bottleneck. This study provides an honest, empirical baseline for QML in QKD monitoring without making unfounded claims of quantum advantage."*

---

## Part B: The 3–5 Minute Technical Presentation

### 1. Motivation and Problem Context
- In discrete-variable decoy-state BB84 QKD, sophisticated side-channel attacks (such as photon-number splitting and detector blinding) manipulate multi-photon yields and detector click correlations while keeping QBER within nominal bounds ($1.5\%–3.5\%$).
- Classical ML models work well, but explore whether quantum state representations in Hilbert space offer superior non-linear boundaries in low-data regimes.

### 2. Implemented Models
We developed three distinct quantum architectures on 4 qubits using PennyLane's `default.qubit` simulator:
1. **Hybrid Variational Quantum Classifier (VQC):** $R_Y$ angle embedding $\to$ 2 layers of Strongly Entangling parameterized rotations with CNOT ring entanglement (24 quantum parameters) $\to$ Pauli-$Z$ expectation measurements on all 4 wires $\to$ trainable classical linear head with Sigmoid activation.
2. **Quantum Kernel SVM (QSVM):** Quantum feature map $U_{\text{map}}(\mathbf{x})$ evaluating a fidelity-based Gram matrix $K(\mathbf{x}, \mathbf{x}') = |\langle \phi(\mathbf{x}) | \phi(\mathbf{x}') \rangle|^2$ $\to$ Scikit-learn dual SVM optimization with validation-calibrated penalty $C=1.0$.
3. **Unsupervised Quantum Autoencoder (QAE):** 4 qubits compressed into 2 latent wires, trained purely on normal telemetry to maximize the projection fidelity of the 2 trash wires onto the reference vacuum state $|00\rangle$.

### 3. Data Representation and Anti-Leakage Protocol
- Consumes `data/v0.5_small/v05_small_dataset` (100 windows across 20 physical scenarios).
- Features reduced from 48 to 4 components using train-only PCA (preserving $64.3\%$ variance) and scaled to $[0, \pi]$.
- Partitions enforced at the scenario level: Train ($N=30$, 6 scenarios), Val ($N=10$, 2 scenarios), Test ($N=20$, 4 held-out scenarios), Generalization ($N=20$, 4 OOD scenarios). Zero scenario overlap across splits.

### 4. Key Empirical Findings
- **VQC and QAE Performance:** Both achieved $\text{Test Accuracy} = 75.0\%$, $\text{Precision} = 71.4\%$, $\text{Recall} = 62.5\%$, $\text{F1} = 0.667$, and $\text{AUROC} = 0.688$ on held-out test data.
- **Comparison with Classical Baselines on the Exact Same 4 Features:**
  - A small Classical MLP (4 $\to$ 8 $\to$ 1, 49 parameters) collapsed to predicting all zeros ($\text{Recall} = 0.0\%$, $\text{F1} = 0.0\%$).
  - Classical RBF-SVM was overly conservative ($\text{Recall} = 12.5\%$, $\text{F1} = 0.200$).
  - Classical Logistic Regression had $\text{F1} = 0.348$.
  - *Conclusion:* VQC and QAE provided effective regularization in the small-sample regime.
- **Quantum Kernel SVM:** Detected $100.0\%$ of attacks ($8/8$ on test, $11/12$ on generalization), but suffered low specificity ($8.3\%$) due to kernel matrix concentration.
- **Computational Efficiency:** VQC trained in $1.01\,\text{s}$, QSVM Gram matrix built in $0.80\,\text{s}$, per-sample inference took $<0.5\,\text{ms}$.

### 5. Academic Defense and Limitations
- QML does **not** outperform the full 48-feature classical pipeline (Robust MCD achieved $\text{F1} = 0.750$).
- The 4-qubit constraint creates an information bottleneck.
- Simulations were executed on classical CPUs; no claim of quantum hardware speedup is made.

---

## Part C: Mathematical Formulations for Defense

### 1. VQC State Preparation and Parameterized Unitary
$$\mathbf{x} \in [0, \pi]^4 \implies |\psi(\mathbf{x})\rangle = \bigotimes_{i=0}^3 R_Y(x_i) |0000\rangle, \quad \text{where } R_Y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$
$$|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle = U_{\text{entangle}}^{(2)} U_{\text{rot}}^{(2)}(\boldsymbol{\theta}_2) U_{\text{entangle}}^{(1)} U_{\text{rot}}^{(1)}(\boldsymbol{\theta}_1) |\psi(\mathbf{x})\rangle$$
$$z_i = \langle \psi(\mathbf{x}, \boldsymbol{\theta}) | Z_i | \psi(\mathbf{x}, \boldsymbol{\theta}) \rangle \in [-1, 1], \quad p(y=1|\mathbf{x}) = \sigma\left(\sum_{i=0}^3 w_i z_i + b\right)$$

### 2. Quantum Fidelity Kernel
$$K(\mathbf{x}, \mathbf{x}') = |\langle \phi(\mathbf{x}) | \phi(\mathbf{x}') \rangle|^2 = \text{Tr}\left( |\phi(\mathbf{x})\rangle\langle\phi(\mathbf{x})| \cdot |\phi(\mathbf{x}')\rangle\langle\phi(\mathbf{x}')| \right) = P(|0000\rangle \text{ under } U_{\text{map}}^\dagger(\mathbf{x}') U_{\text{map}}(\mathbf{x}))$$
- Guaranteed properties: $K_{ii} = 1.0$, $K_{ij} = K_{ji}$, and $\boldsymbol{\alpha}^T K \boldsymbol{\alpha} \ge 0$ (positive semi-definite).

### 3. Quantum Autoencoder Compression Objective
$$\text{Wires: } \mathcal{H}_{\text{total}} = \mathcal{H}_{\text{latent}} \otimes \mathcal{H}_{\text{trash}} = \mathbb{C}^4 \otimes \mathbb{C}^4$$
$$F(\boldsymbol{\theta}, \mathbf{x}) = \text{Tr}_{\text{trash}}\left( \langle 00|_{\text{trash}} \rho_{\text{out}} |00\rangle_{\text{trash}} \right) = P(\text{trash wires } = |00\rangle)$$
$$\text{Anomaly Score: } s(\mathbf{x}) = 1.0 - F(\boldsymbol{\theta}, \mathbf{x}) \in [0, 1]$$

---

## Part D: Answers to Likely Review Panel Questions

### Q1: Why are you using Quantum Machine Learning for QKD anomaly detection?
> **Answer:** *"QKD physical infrastructure operates on quantum states, but its management plane produces classical telemetry (click rates, QBER, yields). Exploring QML allows us to investigate whether encoding continuous telemetry into quantum Hilbert space provides geometric advantages or non-linear decision boundaries that generalize better in low-sample regimes than classical models."*

### Q2: Why not use only classical machine learning?
> **Answer:** *"Classical ML is effective and serves as our primary production pipeline in V1.0. However, Review 2 specifically tasked us with evaluating whether quantum models offer practical utility or complementary detection capabilities. Our goal was not to replace classical ML, but to conduct an honest, empirical benchmarking study."*

### Q3: What exactly is 'quantum' about the VQC?
> **Answer:** *"The feature space is mapped into a 16-dimensional complex projective Hilbert space ($\mathbb{C}^{16}$) via 4 qubits. Entangling CNOT gates create non-local quantum correlations (entanglement) across the telemetry dimensions, allowing parameterized rotation gates to explore transformations that classical linear models cannot represent."*

### Q4: What is a variational quantum circuit?
> **Answer:** *"A variational quantum circuit (VQC) is the quantum analogue of a neural network layer. It consists of fixed state-preparation gates followed by parameterized unitary gates ($R_X, R_Y, R_Z$) and entangling gates. Its parameters $\boldsymbol{\theta}$ are optimized classically via gradient descent to minimize a loss function."*

### Q5: How are classical telemetry features encoded into qubits?
> **Answer:** *"We use Angle Embedding: after scaling our 4 telemetry components to the interval $[0, \pi]$, each feature $x_i$ serves as the rotation angle for a Pauli-$Y$ rotation gate $R_Y(x_i)$ applied to qubit $i$, mapping the zero state $|0\rangle$ to $\cos(x_i/2)|0\rangle + \sin(x_i/2)|1\rangle$."*

### Q6: What do the measured expectation values represent?
> **Answer:** *"The observables are Pauli-$Z$ operators measured on each of the 4 wires: $\langle Z_i \rangle = P(|0\rangle) - P(|1\rangle) \in [-1, 1]$. They quantify the net polarization of each qubit along the computational basis after the variational unitary has acted on the encoded telemetry."*

### Q7: How does the model learn?
> **Answer:** *"We use hybrid quantum-classical optimization. In the forward pass, the quantum simulator evaluates the expectation values, which pass through a classical linear layer and sigmoid activation to produce a probability. We compute Binary Cross-Entropy loss and use PyTorch's automatic differentiation with parameter-shift backpropagation to update both the 24 quantum circuit angles and the 5 classical weights using Adam."*

### Q8: What is the role of the classical optimizer?
> **Answer:** *"The classical optimizer (Adam, $\eta = 0.02$) computes parameter updates outside the quantum circuit. It iteratively adjusts the quantum rotation angles $\boldsymbol{\theta}$ to minimize the classification error, navigating the cost landscape."*

### Q9: What is a quantum kernel?
> **Answer:** *"A quantum kernel is a measure of similarity between two classical data points $\mathbf{x}$ and $\mathbf{x}'$ computed as the transition probability (fidelity) between their quantum feature states: $K(\mathbf{x}, \mathbf{x}') = |\langle \phi(\mathbf{x}) | \phi(\mathbf{x}') \rangle|^2$. It implicitly computes inner products in a high-dimensional Hilbert space without explicitly evaluating coordinates."*

### Q10: How does the Quantum Kernel SVM differ from a classical RBF SVM?
> **Answer:** *"A classical RBF SVM measures similarity via Euclidean distance in input space ($e^{-\gamma \|\mathbf{x} - \mathbf{x}'\|^2}$). The Quantum Kernel SVM maps data into quantum states using entangling unitaries and measures Hilbert-space fidelity. Non-classical entangling operations allow the kernel to capture multi-variable correlations that cannot be factored into pairwise Euclidean distances."*

### Q11: Why did you select only 4 features?
> **Answer:** *"Simulating $n$ qubits on classical computers scales exponentially ($\mathcal{O}(2^n)$ in memory and state operations). Using 4 qubits (16-dimensional state vector) allowed fast, stable training ($<1.5\,\text{s}$) while avoiding barren plateaus (vanishing gradients). We compressed the 48 features using train-only PCA, retaining $64.3\%$ of the variance."*

### Q12: How many qubits did you use and why?
> **Answer:** *"We used 4 qubits across all three architectures. Four qubits represent the optimal balance between expressibility (16-dimensional Hilbert space) and computational tractability for reproducible university review defense."*

### Q13: How do you avoid data leakage?
> **Answer:** *"We enforce strict scenario-level partitioning: all windows from a single scenario reside exclusively in Train, Val, Test, or Generalization. Preprocessing medians, scalers, and PCA matrices are fitted strictly on the Train partition. Thresholds are calibrated strictly on Val. Test and Generalization partitions remain completely unseen until final evaluation."*

### Q14: What were the actual results?
> **Answer:** *"On the held-out IID test set: VQC achieved Accuracy = 75.0%, Precision = 71.4%, Recall = 62.5%, F1 = 0.667, AUROC = 0.688, and AUPRC = 0.710. The Quantum Autoencoder achieved identical F1 = 0.667 with AUPRC = 0.744. The Quantum Kernel SVM achieved 100.0% recall (8/8 attacks detected) with F1 = 0.593."*

### Q15: Which model performed best and according to which metric?
> **Answer:** *"The Hybrid VQC and Quantum Autoencoder tied for the best balanced performance ($\text{F1} = 0.6667$, $\text{Accuracy} = 75.0\%$). If the priority is zero false negatives, the Quantum Kernel SVM performed best on recall ($100.0\%$), but at the expense of higher false alarms."*

### Q16: Did QML outperform the classical baselines?
> **Answer:** *"It depends on the comparison: On the exact same 4-feature input space, QML significantly outperformed the small classical neural network ($\text{F1} = 0.667$ vs. $0.000$) and classical RBF-SVM ($\text{F1} = 0.667$ vs. $0.200$). However, when compared against our full 48-feature classical pipeline from V0.6, the classical Robust MCD estimator achieved a higher F1 score ($0.750$) because it had access to all 48 telemetry dimensions without the 4-qubit bottleneck."*

### Q17: How long did training and inference take?
> **Answer:** *"Training was extremely fast: VQC trained in 1.01 seconds (35 epochs), the QSVM Gram matrix was computed in 0.80 seconds, and the Quantum Autoencoder trained in 0.34 seconds. Per-sample inference latency measured 0.48 ms for VQC and 0.44 ms for QAE on CPU."*

### Q18: Does using a quantum simulator demonstrate quantum advantage?
> **Answer:** *"No, absolutely not. Running PennyLane on a classical CPU simulator simulates quantum mechanics using classical linear algebra. It proves algorithmic validity and model expressiveness, but cannot demonstrate physical quantum advantage or computational speedup."*

### Q19: What are the limitations of this experiment?
> **Answer:** *"The primary limitations are: (1) The 4-qubit information bottleneck that discards higher-order physical features; (2) The relatively small dataset ($N=30$ train, $N=20$ test); and (3) Evaluation on software-simulated telemetry rather than physical optical fiber testbeds."*

### Q20: How could the work be extended to real QKD infrastructure?
> **Answer:** *"Future work would: (1) Deploy the circuits onto real physical quantum processing units (QPUs) via IBM Quantum or Amazon Braket; (2) Scale qubit count to 8–16 qubits using quantum autoencoders to compress all 48 telemetry channels; and (3) Connect the inference pipeline to real-time optical power meters and single-photon detector DAQ interfaces."*

---
*End of QML Review 2 Presentation Notes and Oral Defense Guide.*
