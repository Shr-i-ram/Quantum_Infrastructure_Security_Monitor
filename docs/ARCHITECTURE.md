# Quantum Infrastructure Security Monitor (QISM) — End-to-End System Architecture

**Project:** Quantum Infrastructure Security Monitor (QISM)  
**Milestone:** Version 1.0 — Hardened, Validated, Reproducible Research Prototype  
**Date:** October 2026  

---

## 1. System Vision & Evolutionary Hierarchy

The Quantum Infrastructure Security Monitor (QISM) is a research-grade security monitoring architecture designed to detect physical, channel, and side-channel eavesdropping attacks on practical Quantum Key Distribution (QKD) optical links.

QISM is structured across nine foundational research milestones:

```
[ V0.1–V0.2 Physical Foundation ]
       Decoy-State BB84 (mu, nu, omega), Fiber Loss, Chromatic Dispersion, SPAD Dark Counts, Dead Time
                                      │
                                      ▼
[ V0.3 Multi-Subsystem Telemetry ]
       Alice Laser Driver, Channel Optical Fiber, Bob Detection Unit, Environmental Thermal Sensors
                                      │
                                      ▼
[ V0.4 Attack & Scenario Modeling ]
       Intercept-Resend, Photon-Number-Splitting, Blinding / Control, Trojan-Horse, Thermal Perturbations
                                      │
                                      ▼
[ V0.5 Dataset Framework ]
       Reproducible Sharded Parquet Generation, Strict Seed Isolation, Zero Ground-Truth Leakage
                                      │
                                      ▼
[ V0.6 Baseline Anomaly Detection ]
       Point ML Models (Isolation Forest, OCSVM, Autoencoders), Dimensionless Normalization, Threshold Calibration
                                      │
                                      ▼
[ V0.7 Temporal & Streaming Anomaly Detection ]
       Causal Sliding Windows (W_t), Rolling Z-Score, LSTM/Sequence AE, Operational Alarm Machine (K-of-N Hysteresis)
                                      │
                                      ▼
[ V0.8 Concept Drift & Safe Adaptive Monitoring ]
       Multi-Metric Drift Tracking (Wasserstein, KS, PSI), Reference Windows, Contamination Gates, Shadow Validation
                                      │
                                      ▼
[ V0.9 Integrated Security Monitoring & Incident Analysis ]
       Multivariate Signal Fusion, Temporal Correlation, Incident Lifecycle, Explainable Severity, Auditable Alerts
                                      │
                                      ▼
[ V1.0 Hardened, Validated Research Prototype ]
       System Hardening, Multi-Seed Statistical Benchmarking, Checkpoint Recovery, Reproducibility Release
```

---

## 2. The Core Monitoring Hierarchy: Telemetry to Alert

A foundational principle of QISM V0.9 is the strict separation of monitoring abstraction layers:

```
+---------------------------------------------------------------------------------------------------+
| 1. TELEMETRY RECORD (Raw/Processed Physical Observation)                                          |
|    X_t = {qber, signal_gain, dark_count_rate, clock_drift_rate, sensor_temperature, ...}          |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 2. DETECTOR SIGNALS (Normalized Evidence from Heterogeneous Subsystems)                           |
|    - Temporal Anomaly Score:  S_anom in [0, 1]                                                    |
|    - Statistical Drift Score: D_drift in [0, 1]                                                   |
|    - Physical Deviations:     z_qber, z_gain, z_timing in [0, 1]                                  |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 3. DETECTOR ALARMS (Threshold & Persistence-Based Transition States)                              |
|    Temporal Alarm State Machine: NORMAL -> SUSPECT -> ALARM -> COOLDOWN                           |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 4. SIGNAL FUSION & CORRELATION (Multivariate Security Evidence Synthesis)                         |
|    Unified Evidence Score: F_t = sum(w_i * s_i) * (1 + gamma * k_persist)                         |
|    Incident Lifecycle Machine: OPENING -> ACTIVE -> ESCALATED -> RESOLVED                         |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 5. STRUCTURED OPERATIONAL ALERT & ADVISORY GUIDANCE                                               |
|    Alert { alert_id, incident_id, severity, evidence, subsystems, machine_explanation, recommendation }|
|    Append-Only JSONL Audit Trail: Every state transition reproducibly logged.                     |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Mathematical Formulation of the V0.9 Pipeline

Let $X_t \in \mathbb{R}^d$ be the vector of observable physical telemetry at aggregation step $t$.

### 3.1 Causal Temporal Windowing
$$W_t = \{X_{t-L+1}, X_{t-L+2}, \dots, X_t\} \in \mathbb{R}^{L \times d}$$
Zero future lookahead is permitted; all inference is strictly causal.

### 3.2 Anomaly & Drift Engines
- **Anomaly Score:** $A_t = f(W_t; \mathcal{M}_t)$, where $\mathcal{M}_t$ is the active anomaly detection model.
- **Drift Distance:** $D_t = g(W_{\text{eval}}, \mathcal{R}_t)$, where $\mathcal{R}_t$ is the active normal reference baseline and $g$ is the normalized 1-Wasserstein distance.

### 3.3 Multivariate Signal Fusion Engine
Heterogeneous physical deviations and detector scores are transformed into normalized dimensionless evidence signals $s_i(t) \in [0, 1]$:
$$s_{\text{anom}}(t) = \min\left(1.0, \frac{A_t}{2 \cdot \tau_{\text{anom}}}\right)$$
$$s_{\text{drift}}(t) = \min\left(1.0, \frac{D_t}{2 \cdot \tau_{\text{drift}}}\right)$$
$$s_{\text{feat}, j}(t) = \min\left(1.0, \frac{|X_{t, j} - \mu_{\text{nom}, j}|}{3 \cdot \sigma_{\text{nom}, j}}\right)$$

The fused evidence score combines weighted contributions with persistence amplification:
$$F_t = \min\left(1.0, \left( \sum_{i=1}^M w_i s_i(t) + \delta_{\text{synergy}} \right) \cdot (1.0 + \beta \min(k_{\text{persist}}, 5)) \right)$$
where $\delta_{\text{synergy}}$ is applied when three or more orthogonal subsystems deviate concurrently.

### 3.4 Incident State Machine & Correlation
An incident $\mathcal{I}$ is initiated when $F_t \ge \tau_{\text{open}}$ or an active alarm occurs:
$$\text{State}_t = \begin{cases}
\text{OPENING}, & \text{if } \mathcal{I} \text{ was None and } F_t \ge \tau_{\text{open}} \\
\text{ACTIVE}, & \text{if } \mathcal{I} \text{ persists for } \ge 2 \text{ windows} \\
\text{ESCALATED}, & \text{if } F_t \ge \tau_{\text{escalate}} \\
\text{RESOLVED}, & \text{if } F_t < \tau_{\text{open}} \text{ for } N_{\text{patience}} \text{ consecutive windows}
\end{cases}$$

Constituent alarms occurring within the temporal correlation horizon $T_{\text{corr}}$ are merged into $\mathcal{I}$, compressing redundant alerts and preventing operator fatigue.

---

## 4. Security Boundaries & Zero Ground-Truth Leakage

QISM strictly enforces an architectural barrier between operational inference and offline scientific evaluation:

```
[ SIMULATION RUNTIME / DATASET ]
  Observable Features (16 channels)      Ground Truth Sidecars (Offline only!)
  - qber, signal_gain, dark_counts...    - attack_active, attack_type, true_loss_delta...
         │                                      │
         ▼                                      │
┌─────────────────────────────────┐             │
│   OPERATIONAL MONITOR PIPELINE  │             │
│   - Schema Validation & Strip   │             │
│   - Temporal Anomaly Detector   │             │
│   - Statistical Drift Detector  │             │
│   - Safe Adaptation Gate        │             │
│   - Signal Fusion Engine        │             │
│   - Incident Lifecycle Manager  │             │
│   - Deterministic Severity      │             │
│   - Structured Alerts & Audit   │             │
└────────────────┬────────────────┘             │
                 │                              │
                 ▼ Inferred Incidents/Alerts    │
┌───────────────────────────────────────────────▼─┐
│        OFFLINE EVALUATION & AUDIT LAYER         │
│  - Incident Precision, Recall, F1               │
│  - Detection Latency & Alert Latency            │
│  - Alert Compression Ratio                      │
└─────────────────────────────────────────────────┘
```

1. **Strict Sanitization:** Any incoming records containing ground-truth tags (`attack_type`, `attack_active`, `ground_truth_*`) have these fields stripped prior to reaching the ML or fusion engines.
2. **Contamination Gating:** If $A_t \ge \tau_{\text{safe}}$ or an alarm is active, all background adaptation candidate gathering is aborted. Malicious telemetry is never incorporated into normal reference distributions.
3. **No Automatic Physical Responses:** The system emits structured alerts and advisory recommendations (`is_automated_execution = False`). It never modifies hardware or optical attenuation.
