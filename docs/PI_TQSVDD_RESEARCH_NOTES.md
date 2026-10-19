# QISM PI-TQSVDD — Research and Review Notes
## Theoretical Foundations, Prior-Art Differentiation, and Defense Guidance

---

### Executive Research Metadata
- **Project:** Quantum Infrastructure Security Monitor (QISM)
- **Document:** Research Notes and Defense Analysis (`docs/PI_TQSVDD_RESEARCH_NOTES.md`)
- **Author:** QISM Research & Development Team
- **Baseline Version:** V1.0 Hardened Architecture
- **Audit Date:** Current Baseline (October 2026)

---

## 1. Prior-Art Analysis: QSVDD and NQSVDD

### 1.1 Classical Deep SVDD (Ruff et al., 2018)
- **Concept:** Classical deep neural network maps inputs to a minimum-volume hypersphere in output feature space:
  $$\min_{\mathcal{W}} \frac{1}{N} \sum_{i=1}^N \|\phi(\mathbf{x}_i; \mathcal{W}) - \mathbf{c}\|^2 + \frac{\lambda}{2} \sum_{l=1}^L \|\mathbf{W}_l\|_F^2$$
- **Representation Collapse Phenomenon:** If bias terms are included in every layer, or if all weights decay to zero ($\mathbf{W} \to \mathbf{0}$), the network trivially collapses all inputs to a constant vector: $\phi(\mathbf{x}) = \mathbf{c}$, producing zero loss for all inputs.
- **Ruff et al. Mitigation:** Remove bias terms in all layers, initialize weights away from zero, and fix the centre $\mathbf{c} \neq \mathbf{0}$ as a non-trainable reference.

### 1.2 Quantum SVDD (QSVDD)
- **Concept:** Uses a parameterized quantum circuit (PQC) as the feature extractor $U(\boldsymbol{\theta})$ to map data into Hilbert space states $|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle$.
- **Hypersphere Formulation:** Measures expectation values $\mathbf{z} = \langle \psi | \mathbf{M} | \psi \rangle$ and penalizes distance $\|\mathbf{z} - \mathbf{c}\|^2$.
- **Prior Limitations in Published Literature:**
  - Evaluated almost exclusively on static toy benchmarks (e.g. Iris, MNIST-2 classes).
  - Uses basic single-layer angle encoding, creating an severe information bottleneck when the input dimension exceeds the small qubit count ($D > n$).
  - Ignores temporal dependencies: each observation is treated as an independent snapshot.

### 1.3 Neural Quantum SVDD (NQSVDD Preprint)
- **Concept:** Hybrid classical-quantum deep SVDD combining classical feature extraction layers with quantum variational circuits.
- **Differences from PI-TQSVDD:**
  1. *Temporal Dynamics:* NQSVDD is fundamentally static; it does not model sequence evolution or transition dynamics. PI-TQSVDD explicitly introduces a causal temporal prediction objective $\mathcal{L}_{\text{temporal}} = \|\mathbf{z}_{t+1} - \hat{\mathbf{z}}_{t+1}\|^2$ directly on the quantum latent representations.
  2. *Domain Grounding:* PI-TQSVDD is physics-informed by QKD physical layer telemetry (discrete-variable BB84 decoy states, detector afterpulsing, and channel transmittance drift).
  3. *Multi-Sensor Telemetry Preservation:* PI-TQSVDD introduces physical domain feature-group partitioning and multi-layer data reuploading, resolving the 48-feature bottleneck.

---

## 2. Theoretical Contributions of PI-TQSVDD

1. **Temporal Quantum Representation Learning:**
   - Demonstrates that tracking temporal residuals over quantum latent states ($\hat{\mathbf{z}} = g_{\boldsymbol{\psi}}(\mathbf{Z})$) provides a +21.3 percentage point increase in test AUROC over static Quantum SVDD. Eavesdropping attacks are physical processes that cause transitions across operating states; static spatial distance alone is insufficient.
2. **Mitigation of the Dimensionality-Qubit Mismatch:**
   - Demonstrates that **Data Reuploading** and **Learned Projections** allow a 4-qubit circuit to effectively consume information from 48 continuous telemetry channels without the performance degradation suffered by basic angle encoding or amplitude normalization.
3. **True One-Class Integrity:**
   - Validates that legitimate QKD security monitoring can be achieved with zero attack exposure during training ($F_1 = 0.67 - 0.70$), outperforming Classical Deep SVDD ($F_1 = 0.625$, $\text{AUROC} = 0.612$) while avoiding the unrealistic supervised assumptions of previous VQC prototypes.

---

## 3. Defense Guidance for Review 2 & Research Reviews

### Defensible Scientific Claims (Supported by Empirical Artifacts)
- *"PI-TQSVDD is a legitimate, authentic one-class quantum anomaly detector trained strictly without attack labels."* (Supported by `artifacts/pi_tqsvdd/experiment_registry.csv` and `test_data.py`).
- *"Causal temporal prediction over quantum latent representations significantly outperforms static quantum SVDD (+21.3% AUROC on test partitions)."* (Supported by `pilot_4q_reupload` vs `pilot_4q_static_qsvdd`).
- *"Data Reuploading and Learned Projections resolve the information bottleneck of encoding 48 physical channels into 4 qubits, outperforming basic angle encoding by 16.7% in F1 score."* (Supported by `pilot_4q_reupload` vs `pilot_4q_angle`).
- *"Among one-class models, PI-TQSVDD outperforms Classical Deep SVDD, Isolation Forest, and One-Class SVM on both F1 and AUROC."* (Supported by `artifacts/pi_tqsvdd/figures/07_model_family_comparison.png`).

### Premature Claims to Strictly Avoid
- **DO NOT claim "Quantum Advantage" or "Quantum Supremacy":** The experiments were executed on classical simulation hardware (`lightning.qubit`). The goal is architectural feasibility and representation quality, not asymptotic runtime supremacy.
- **DO NOT claim Superiority over Supervised VQC:** The supervised VQC had direct access to attack labels during training and achieved higher test recall ($0.800$ vs $0.583$). Acknowledge that supervised classifiers hold an unfair advantage on known attacks, but fail completely on zero-day attacks where PI-TQSVDD remains operable.
- **DO NOT claim Scalability to 50+ Qubits:** The empirical sweet spot is currently **4 to 6 qubits**; 8 qubits exhibited diminishing returns and slight overfitting in the low-sample training regime.
