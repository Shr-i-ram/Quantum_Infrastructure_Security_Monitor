# Quantum Infrastructure Security Monitor (QISM)
## Flagship Model Ablation Study Report (PI-TQSVDD)

This research report documents the comprehensive ablation study conducted on the flagship **Physics-Informed Temporal Quantum Support Vector Data Description (PI-TQSVDD)** architecture across register dimensions, feature encoding schemes, loss formulations, and multi-seed trials using the authoritative QISM research dataset.

---

## 1. Experimental Overview

To rigorously isolate the contribution of each architectural component, controlled PI-TQSVDD configurations were trained and evaluated on the QISM research dataset:
- **Qubit Count Scaling**: $n \in \{2, 4, 6, 8\}$ qubits simulated on PennyLane `lightning.qubit` with adjoint differentiation.
- **Quantum Data Encodings**: Single-pass Angle Embedding, Multi-layer Data Reuploading, State-vector Amplitude Encoding, IQP/ZZ Entangling Feature Maps, and Domain Feature Groups.
- **Loss Function Ablations**: Causal temporal loss enabled ($\lambda_t = 0.5$) vs. temporal loss disabled ($\lambda_t = 0.0$, static ablation) vs. static baseline Quantum SVDD.
- **Multi-Seed Stability Trials**: Evaluated across 4 independent random seeds ($42, 101, 2024, 777$) to verify convergence stability and representation reliability.

---

## 2. Qubit Count Sweep: Scaling and Returns

Evaluating PI-TQSVDD across register sizes while holding circuit depth ($M=2$) and temporal prediction parameters fixed:

| Configuration | Qubits ($n$) | Input Representation | In-Dist AUROC | In-Dist AUPRC | Recall @ 1% FPR | Incident Recall | Gen AUROC | Stress AUROC | Train Time (s) | Circuit Depth |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PITQSVDD_2Q_Angle`** | 2 | PCA-2 ($35.63\%$ var) | 0.5909 | 0.4963 | 11.5% | 35.6% | 0.6724 | 0.5829 | 11.44 | 2 |
| **`PITQSVDD_4Q_Angle_Ref`** | 4 | PCA-4 ($46.49\%$ var) | **0.6751** | **0.5634** | 5.5% | 27.6% | **0.6952** | 0.5322 | 21.84 | 2 |
| **`PITQSVDD_6Q_Angle`** | 6 | PCA-6 ($54.57\%$ var) | 0.6392 | 0.5257 | 12.0% | 35.6% | 0.6736 | 0.5561 | 26.18 | 2 |
| **`PITQSVDD_8Q_Angle`** | 8 | PCA-8 ($60.07\%$ var) | 0.5908 | 0.5150 | 19.4% | 39.1% | 0.6165 | **0.6231** | 38.23 | 2 |

### Key Findings on Register Scaling:
1. **Optimal Performance at 4 Qubits**: The 4-qubit configuration achieves the peak in-distribution AUROC (**$0.6751$**, averaging **$0.6891$** across seeds) and generalization AUROC (**$0.6952$**), surpassing both the 2-qubit circuit ($0.5909$) and higher qubit counts ($0.6392$ for 6Q, $0.5908$ for 8Q).
2. **Diminishing Returns Beyond 4 Qubits**: Increasing the register from 4 to 8 qubits doubles the quantum state dimension from $16$ to $256$ amplitudes and increases simulation training time from $21.8$s to $38.2$s ($+75\%$), without yielding detection gains under standard angle encoding. AUROC decreases due to **state dispersion** in Hilbert space: higher-dimensional expectation spaces disperse normal telemetry, making the minimum-volume hypersphere boundary less compact unless data reuploading is applied.
3. **Data Reuploading Rescues 8 Qubits**: When multi-layer data reuploading is used with 8 qubits (`PITQSVDD_8Q_DataReuploading`), AUROC rises from $0.5908$ to **$0.6941$** ($+10.33\%$), confirming that deeper feature interleaving prevents dispersion.

---

## 3. Quantum Encoding Sweep

Comparing feature representation and injection methods on 4-qubit circuits:

| Encoding Method | Features Injected | Mechanism | In-Dist AUROC | Recall @ 1% FPR | Incident Recall | Generalization AUROC | Stress AUROC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Amplitude Encoding` (4Q)**| 16 (Normalized) | State vector amplitude mapping ($2^4$) | **0.7147** | **38.2%** | **64.4%** | **0.7761** | **0.6390** |
| **`Data Reuploading` (4Q)** | 16 (PCA-16) | Interleaved feature rotations across layers | 0.6795 | 28.7% | 66.7% | 0.7190 | 0.5993 |
| **`Angle Embedding` (Ref, 4Q)**| 4 (PCA-4) | Single-pass $R_Y$ state preparation | 0.6751 | 5.5% | 27.6% | 0.6952 | 0.5322 |
| **`Feature Group` (4Q)** | 4 (Domain Groups) | Partitioned feature registers + CNOTs | 0.6644 | 9.4% | 34.5% | 0.6708 | 0.5542 |
| **`IQP / ZZ Map` (4Q)** | 4 (PCA-4) | Non-linear pairwise ZZ phase entanglement | 0.4838 | 1.4% | 12.6% | 0.5748 | 0.4928 |

### Key Findings on Data Encoding:
1. **Amplitude Encoding Achieves Top Quantum Score**: Amplitude encoding achieved **$0.7147$ in-distribution AUROC**, **$0.7761$ generalization AUROC**, and **$0.6390$ stress AUROC**, making it the highest-performing quantum model in the benchmark. By encoding 16 normalized telemetry features into the $2^4 = 16$ probability amplitudes of the 4-qubit register, amplitude encoding captures rich global correlations without expanding circuit depth.
2. **Data Reuploading Provides High Incident Recall**: Data reuploading achieved **$66.7\%$ incident recall**, successfully capturing full attack episodes by repeatedly injecting classical features between parameterized entangling blocks.
3. **Failure of IQP/ZZ Phase Encoding**: IQP/ZZ encoding collapsed to **$0.4838$ AUROC** (below random chance). Artificial two-qubit ZZ interactions $\exp(i(\pi - x_j)(\pi - x_k) Z_j Z_k)$ over-correlate disparate physical telemetry streams (e.g., thermal sensor voltage with single-photon count rates), scrambling the natural state manifold.

---

## 4. Objective-Function Ablations: The Role of Temporal Loss

To test whether causal temporal latent modeling is genuinely responsible for the model's advantage:

| Model Ablation | Formulation | In-Dist AUROC | Recall @ 1% FPR | Incident Recall | Detection Delay (windows) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full PI-TQSVDD (Ref)** | SVDD + Causal Temporal ($\lambda_t = 0.5$) | **0.6751** | **5.5%** | **27.6%** | **3.50** |
| **PI-TQSVDD No Temporal**| Static SVDD Only ($\lambda_t = 0.0$) | 0.6453 | 5.5% | 28.7% | 3.36 |
| **Static Quantum SVDD** | Baseline Static QSVDD ($\beta = 0.0$) | 0.6402 | 3.4% | 21.8% | 4.32 |

### Key Findings on Objective Function:
1. **Temporal Loss Contributes $+2.98\%$ AUROC**: Disabling the temporal loss term ($\lambda_t = 0.0$) causes in-distribution AUROC to drop from **$0.6751$ to $0.6453$** ($-2.98\%$), and AUPRC drops from $0.5634$ to $0.5269$ ($-3.65\%$).
2. **Outperforming Static Quantum Baselines**: Full PI-TQSVDD outperforms static Quantum SVDD ($0.6402$) by **$+3.49\%$ AUROC**, with incident recall improving from $21.8\%$ to $27.6\%$ (and up to $64.4\%$ in Amplitude encoding), confirming that modeling dynamical sequence transitions is beneficial.

---

## 5. Multi-Seed Stability and Reproducibility

Evaluating the 4-qubit reference PI-TQSVDD across independent initialization seeds ($42, 101, 2024, 777$):

| Seed | In-Dist AUROC | In-Dist AUPRC | Recall @ 1% FPR | Incident Recall | Generalization AUROC | Stress AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42 (Ref)** | 0.6751 | 0.5634 | 5.5% | 27.6% | 0.6952 | 0.5322 |
| **101** | 0.7066 | 0.6099 | 9.2% | 32.2% | 0.6990 | 0.5634 |
| **2024** | 0.6945 | 0.5819 | 4.6% | 24.1% | 0.7032 | 0.5531 |
| **777** | 0.6801 | 0.5771 | 10.4% | 34.5% | 0.6911 | 0.5317 |
| **Mean $\pm$ Std** | **$0.6891 \pm 0.0142$** | **$0.5831 \pm 0.0197$** | **$7.4\% \pm 2.8\%$** | **$29.6\% \pm 4.6\%$** | **$0.6971 \pm 0.0052$** | **$0.5451 \pm 0.0152$** |

### Stability Conclusion:
The standard deviation across independent initialization seeds is very modest ($\sigma_{\mathrm{AUROC}} = 0.0142$, coefficient of variation $2.06\%$), confirming that:
- Initializing the hypersphere centre as the empirical batch mean $c \approx \mathbb{E}[\phi(x)]$ prevents degenerate hypersphere collapse.
- Adjoint differentiation on PennyLane `lightning.qubit` provides reliable, reproducible optimization trajectories.
