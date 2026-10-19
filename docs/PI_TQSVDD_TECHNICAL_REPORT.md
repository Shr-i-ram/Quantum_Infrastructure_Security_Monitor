# Quantum Infrastructure Security Monitor (QISM)
## Flagship Model Technical Report: Physics-Informed Temporal Quantum SVDD (PI-TQSVDD)

This technical report provides the comprehensive mathematical, architectural, and operational specification of the QISM flagship research model: **Physics-Informed Temporal Quantum Support Vector Data Description (PI-TQSVDD)**.

---

## 1. Architectural Concept and Design Rationale

Quantum Key Distribution (QKD) infrastructure generates continuous, high-dimensional telemetry streams characterized by:
1. **Complex Cross-Feature Correlated Physics**: Subtle optical interactions between photon statistics, channel loss, detector timing jitter, and decoy-state yields.
2. **Causal Temporal Dependencies**: Strong auto-correlations across consecutive time intervals ($1$ ms windows), where attack onset or eavesdropping disrupts physical transition dynamics.
3. **Severe Attack Scarcity**: Training must occur predominantly or exclusively under benign normal operating conditions, as novel zero-day attacks or stealthy quantum hacking strategies cannot be anticipated in advance.

**PI-TQSVDD** addresses these challenges by uniting **variational quantum state representation learning** with **causal recurrent temporal transition modeling** in a one-class support vector data description framework.

```
       CAUSAL TELEMETRY WINDOW: X_t = [x_{t-2}, x_{t-1}, x_t]
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
QUANTUM STATE ENCODING                  VARIATIONAL UNITARY
|psi(x_tau)> = U_enc(x_tau)|0>^n       U_var(theta) with StronglyEntanglingLayers
       │                                       │
       └───────────────────┬───────────────────┘
                           ▼
          QUANTUM OBSERVABLES MEASUREMENT (Pauli-Z)
             z_tau = <psi(x_tau, theta)| Z |psi(x_tau, theta)>
                           │
         LATENT QUANTUM SEQUENCE: Z_t = [z_{t-2}, z_{t-1}, z_t]
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
STATIC QUANTUM SVDD                     CAUSAL TEMPORAL PREDICTOR
||z_t - c||_2^2                         hat{z}_{t+1} = g_psi(Z_t)
       │                                       │
       └───────────────────┬───────────────────┘
                           ▼
        COMBINED CONTINUOUS ANOMALY SCORE: s_t = alpha * s_normal + beta * s_temporal
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Quantum State Preparation and Parameterized Evolution
Given a classical normalized telemetry vector $v_\tau \in [0, \pi]^d$ at time step $\tau \in \{t-L+1, \dots, t\}$ (with causal window length $L = 3$):
1. **State Preparation**: Classical data is injected into an $n$-qubit register using a unitary encoding circuit $U_{\mathrm{enc}}(v_\tau)$:
   $$|\psi_0(v_\tau)\rangle = U_{\mathrm{enc}}(v_\tau) |0\rangle^{\otimes n}$$
   Under standard Angle Embedding:
   $$U_{\mathrm{enc}}(v_\tau) = \bigotimes_{j=0}^{n-1} R_Y(v_{\tau, j})$$
2. **Parameterized Variational Unitary**: The encoded state is transformed by an $M$-layer parameterized ansatz:
   $$|\psi(v_\tau, \theta)\rangle = \left( \prod_{l=1}^M W_l(\theta_l) \right) |\psi_0(v_\tau)\rangle$$
   where each layer $l$ comprises single-qubit rotations followed by entangling gates:
   $$W_l(\theta_l) = U_{\mathrm{ent}} \cdot \bigotimes_{j=0}^{n-1} R_Z(\theta_{l, j, 2}) R_Y(\theta_{l, j, 1}) R_X(\theta_{l, j, 0})$$
   and $U_{\mathrm{ent}}$ applies periodic CNOT gates: $\mathrm{CNOT}(j, (j+1)\%n)$.
3. **Observables Measurement**: The latent quantum representation $z_\tau \in [-1, 1]^n$ is obtained by measuring the expectation value of the Pauli-$Z$ operator on each wire:
   $$z_{\tau, j} = \langle \psi(v_\tau, \theta) | Z_j | \psi(v_\tau, \theta) \rangle \quad \text{for } j \in \{0, \dots, n-1\}$$

Evaluating across the causal window yields the quantum latent trajectory:
$$Z_t = [z_{t-L+1}, \dots, z_t] \in \mathbb{R}^{L \times n}$$

---

### 2.2 SVDD Normality Learning and Centre Initialization
PI-TQSVDD encloses normal quantum state representations within a minimum-volume hypersphere in the observable expectation space $\mathbb{R}^n$.

1. **Normality Centre Initialization**:
   To prevent **representation collapse** (where the circuit parameters collapse all inputs to an identical trivial state), the hypersphere centre $c \in \mathbb{R}^n$ is computed strictly from an initial forward pass over normal training representations:
   $$c_j = \frac{1}{N_{\mathrm{init}}} \sum_{i=1}^{N_{\mathrm{init}}} z_{i, t, j}$$
   If any coordinate $|c_j| < 10^{-4}$, it is perturbed to $c_j = \epsilon = 0.1$ to ensure non-zero gradient flow.
2. **Static SVDD Normality Loss**:
   $$\mathcal{L}_{\mathrm{SVDD}} = \frac{1}{N} \sum_{i=1}^N \|z_{i, t} - c\|_2^2$$

---

### 2.3 Causal Latent Temporal Dynamics
Unlike conventional static SVDD, PI-TQSVDD models the temporal transition dynamics between consecutive quantum latent states using a causal predictor $g_\psi$:
$$\hat{z}_{i, t+1} = g_\psi(Z_{i, t})$$
where $g_\psi$ is a recurrent GRU network followed by a linear projection.

The **Temporal Prediction Loss** penalizes dynamical deviations between predicted and true next quantum states:
$$\mathcal{L}_{\mathrm{temporal}} = \frac{1}{N_{\mathrm{valid}}} \sum_{i \in \mathrm{valid}} \|z_{i, t+1} - \hat{z}_{i, t+1}\|_2^2$$
where $i \in \mathrm{valid}$ denotes windows whose next step does not cross scenario boundaries.

---

### 2.4 Total Multi-Objective Loss Function
The complete joint optimization objective is:
$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{SVDD}} + \lambda_t \mathcal{L}_{\mathrm{temporal}} + \lambda_{\mathrm{reg}} \left( \|\theta\|_2^2 + \|\psi\|_2^2 \right)$$
where:
- $\lambda_t \ge 0$ is the temporal transition weight (ablated in Phase VI).
- $\lambda_{\mathrm{reg}} = 10^{-4}$ provides $L_2$ weight regularization on both variational quantum gate parameters and classical recurrent weights.

---

### 2.5 Unified Continuous Anomaly Scoring
At test time, continuous anomaly scores combine static hypersphere distance and temporal transition error:
$$s_{\mathrm{normal}}(t) = \|z_t - c\|_2^2$$
$$s_{\mathrm{temporal}}(t) = \|z_t - \hat{z}_t\|_2^2$$
$$s_t = \alpha \cdot s_{\mathrm{normal}}(t) + \beta \cdot s_{\mathrm{temporal}}(t)$$
with operational weighting $\alpha = 0.5$ and $\beta = 0.5$.

---

## 3. High-Performance Quantum Simulation Specification

- **Backend**: PennyLane `lightning.qubit` (C++ high-performance state-vector simulator).
- **Differentiation**: Exact **adjoint differentiation** (`diff_method="adjoint"`), avoiding expensive numerical finite-difference and parameter-shift overhead.
- **Hardware Accounting**:
  - 2 Qubits: 6 CNOTs, 12 single-qubit rotations, $12$ trainable quantum parameters.
  - 4 Qubits: 8 CNOTs, 24 single-qubit rotations, $24$ trainable quantum parameters.
  - 6 Qubits: 12 CNOTs, 36 single-qubit rotations, $36$ trainable quantum parameters.
  - 8 Qubits: 16 CNOTs, 48 single-qubit rotations, $48$ trainable quantum parameters.

---

## 4. Anti-Collapse and Training Integrity Verification

PI-TQSVDD incorporates four safeguards against degenerate optimization:
1. **Frozen Normality Centre**: Centre $c$ is fixed after initialization, preventing the optimizer from trivially tracking drifting representations.
2. **Bias-Free Classical Bottlenecks**: When learned classical projections are utilized, layers omit additive bias terms to eliminate constant-output solutions.
3. **Weight Initializations**: Quantum weights are initialized randomly away from zero: $\theta \sim \mathcal{N}(0.1, 0.2^2)$.
4. **Normality Variance Diagnostics**: Verification confirms that the latent variance $\mathrm{Var}(z_{\mathrm{normal}}) > 0.02$, ensuring diverse, expressive state mappings.
