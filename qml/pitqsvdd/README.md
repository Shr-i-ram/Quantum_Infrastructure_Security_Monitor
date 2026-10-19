# Physics-Informed Temporal Quantum Support Vector Data Description (PI-TQSVDD)
### Quantum Infrastructure Security Monitor (QISM)

---

## 1. Overview and Scientific Motivation

**PI-TQSVDD** is a quantum machine learning (QML) architecture designed for physical-layer attack anomaly detection in Quantum Key Distribution (QKD) infrastructure.

While classical machine learning models in QISM (such as Dense Autoencoders, Isolation Forests, and Robust MCD) rely on linear and neural projections over discrete-variable BB84 telemetry, and early QML prototypes (such as supervised Variational Quantum Classifiers and Quantum Kernel SVMs) evaluated 4-feature toy representations under supervised assumptions, **PI-TQSVDD represents a dedicated physical one-class architecture**:

1. **One-Class Normality Objective:** Learns exclusively from legitimate, non-attack QKD operating data by mapping normal telemetry into a compact quantum hypersphere $\mathcal{B}(c, R)$. Training strictly requires zero attack labels.
2. **Causal Temporal Modeling:** Telemetry observations are not independent identically distributed (I.I.D.) static points; they form physical trajectories in time. PI-TQSVDD combines quantum hypersphere normality with a causal temporal prediction residual loss $\mathcal{L}_{\text{temporal}} = \|z_t - \hat{z}_t\|_2^2$ over quantum latent representations.
3. **Full-Feature Telemetry Preservation:** Rather than arbitrarily truncating the simulator's 48 observable features to a naive 4-feature subset, PI-TQSVDD supports learned classical neural bottlenecks, physical domain feature-group partitioning, and multi-layer data reuploading.
4. **Hardware Simulator Scalability:** Fully implemented using PennyLane with the required high-performance `lightning.qubit` state-vector simulator and PyTorch adjoint automatic differentiation across **2, 4, 6, and 8 qubits**.
5. **Production Crash Resilience:** Features atomic two-stage checkpointing (`checkpoint_latest.pt`, `checkpoint_best.pt`) allowing seamless resumption from interruptions without restarting from epoch zero.

---

## 2. Mathematical Formulation

Given a causal telemetry window $\mathbf{X}_t = [\mathbf{x}_{t-L+1}, \dots, \mathbf{x}_t] \in \mathbb{R}^{L \times D}$:

1. **Quantum State Transformation:**
   $$\mathbf{z}_\tau = Q_{\boldsymbol{\theta}}(h_{\boldsymbol{\phi}}(\mathbf{x}_\tau)) \in [-1, 1]^{n_{\text{qubits}}} \quad \text{for } \tau \in [t-L+1, \dots, t]$$
   where $Q_{\boldsymbol{\theta}}$ is a parameterized quantum circuit evaluated via Pauli-$Z$ expectation values on $n$ qubits.

2. **One-Class Hypersphere Loss:**
   $$\mathcal{L}_{\text{SVDD}} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{z}_{i, t} - \mathbf{c}\|_2^2 + \lambda_{\text{reg}} \|\boldsymbol{\theta}\|_2^2$$
   where $\mathbf{c}$ is a fixed reference centre estimated from a preliminary forward pass on normal training data.

3. **Temporal Prediction Objective:**
   $$\hat{\mathbf{z}}_{t+1} = g_{\boldsymbol{\psi}}(\mathbf{z}_{t-L+1}, \dots, \mathbf{z}_t)$$
   $$\mathcal{L}_{\text{temporal}} = \frac{1}{N_{\text{valid}}} \sum_{i \in \text{valid}} \|\mathbf{z}_{i, t+1} - \hat{\mathbf{z}}_{i, t+1}\|_2^2$$

4. **Joint Objective:**
   $$\mathcal{L} = \mathcal{L}_{\text{SVDD}} + \lambda_t \mathcal{L}_{\text{temporal}} + \lambda_{\text{reg}} (\|\boldsymbol{\theta}\|_2^2 + \|\boldsymbol{\psi}\|_2^2)$$

5. **Continuous Anomaly Scoring & Thresholding:**
   $$s_t = \alpha \|\mathbf{z}_t - \mathbf{c}\|_2^2 + \beta \|\mathbf{z}_t - \hat{\mathbf{z}}_t\|_2^2$$
   Decision threshold $\tau$ is calibrated strictly on the 95th percentile of normal validation scores.

---

## 3. Comparison with Existing QML Models

| Architecture | Paradigm | Telemetry Strategy | Qubits | Temporal Dynamics | Attack Label Requirement |
|---|---|---|---|---|---|
| **Variational Quantum Classifier (VQC)** | Supervised Binary | 4-feature PCA truncation | 4Q | None (static) | Required during training |
| **Quantum Kernel SVM (QSVM)** | Kernel Machine | 4-feature PCA truncation | 4Q | None (static) | Required during training |
| **Quantum Autoencoder (QAE)** | Unsupervised Compression | 4-feature PCA truncation | 4Q (2 latent, 2 trash) | None (static) | Normal only |
| **PI-TQSVDD (Flagship)** | **One-Class Hypersphere + Temporal** | **48 Features: Reuploading / Bottleneck / Grouping** | **2, 4, 6, 8Q** | **Causal GRU Residual** | **Normal only (Zero attack contamination)** |

---

## 4. Execution Guide

### Mode A — Smoke Tests (Fast Setup Verification)
To verify circuit execution, gradient computation, and environment functionality on `lightning.qubit`:
```bash
python -m qml.pitqsvdd.train --config qml/pitqsvdd/configs/smoke.yaml --run-id smoke_test --resume auto
```
Or run the automated unit test suite:
```bash
pytest -v qml/pitqsvdd/tests
```

### Mode B — Pilot Benchmark Sweep (Architecture Investigation)
To run the staged pilot benchmark across 2, 4, 6, 8 qubits, all 6 encodings, and classical baselines:
```bash
python qml/pitqsvdd/benchmark_runner.py
```

### Mode C — Definitive Final Training Run
To execute the final full training configuration on the primary dataset:
```bash
python -m qml.pitqsvdd.train --config qml/pitqsvdd/configs/pitqsvdd_final.yaml --run-id pitqsvdd_final --resume auto
```

### Resuming an Interrupted Run
If a training run was halted or interrupted, execute the exact same command with `--resume auto`:
```bash
python -m qml.pitqsvdd.train --config qml/pitqsvdd/configs/pitqsvdd_final.yaml --run-id pitqsvdd_final --resume auto
```
The trainer will detect `checkpoint_latest.pt`, restore all model weights, optimizer state, epoch, and step counts, and continue training seamlessly.

---

## 5. Directory and Artifact Structure

```
qml/pitqsvdd/
    config.py               # Dataclass schemas and YAML loaders
    data.py                 # Normal-only partitioning and data loaders
    feature_transforms.py   # 48-feature PCA, Bottleneck, Grouping, Reuploading
    temporal_windows.py     # Causal temporal window builder
    train.py                # Dedicated CLI entry point
    benchmark_runner.py     # Pilot experiment runner and figure generator
    circuits/               # Parameterized circuits on lightning.qubit
    encodings/              # Angle, Data Reuploading, Amplitude, IQP/ZZ, Feature-Group, Learned Proj
    models/                 # Static QSVDD, Temporal PI-TQSVDD, Context-Aware PI-TQSVDD
    training/               # Resumable trainer, atomic checkpoint manager
    evaluation/             # Scoring, thresholding, point and event-level metrics
    visualization/          # High-resolution plotting utilities
    configs/                # smoke.yaml, pilot.yaml, pitqsvdd_final.yaml
    tests/                  # Complete unit test suite (36 tests)

artifacts/pi_tqsvdd/
    checkpoints/            # Crash-resistant model checkpoints
    figures/                # 300 DPI publication diagnostic plots
    manifests/              # Per-run provenance manifests
    metrics/                # Per-run detailed performance metrics
    experiment_registry.csv # Machine-readable experiment records
```
