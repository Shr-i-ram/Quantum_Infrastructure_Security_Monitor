# QISM Version 0.3 Attack Signatures Report

## Overview
This document characterizes the physical mechanisms, causal telemetry responses, and observable signatures for all 6 attacks implemented in **QISM Version 0.3**.

Each attack operates strictly through physical intervention without directly mutating telemetry features. Telemetry features are computed strictly causally by the aggregation engine.

---

## 1. Attack Signature Matrix

| Attack Type | Physical Intervention | Primary Observable Signature | Secondary Observable Signature | Baseline | Under Attack |
|---|---|---|---|---|---|
| **Intercept-Resend** | Intercepts & resends in Eve basis | **QBER surges** (+20% to +25%) | Key rate collapses; detection rate steady | 0.00% | 35.83% |
| **Channel Loss** | Attenuates quantum channel transmittance | **Signal & decoy gains drop** | Sifted count drops; QBER steady or slight rise | 2.03% | 0.35% |
| **Source Manipulation** | Modifies mean photon number $\mu$ | **Detection rate surges/drops** | Multi-photon fraction alters | 14285.7 Hz | 30700.0 Hz |
| **Detector Manipulation**| Alters target detector efficiency | **Detector click imbalance skew** | Double click rate alters | +0.1459 | -0.6186 |
| **Timing Attack** | Displaces arrival time $\Delta t$ | **Mean timing offset shifts** | Gate edge clipping causes loss | +0.8 ps | +252.6 ps |
| **PNS Attack** | Splits multi-photon & suppresses singles | **Gain ratio anomaly (Signal/Decoy)** | Decoy yield bounds violated | 2799353.69 | 7201313.10 |

---

## 2. Attack Lifecycle & Transition Dynamics
All attacks implement the 5-phase ground truth lifecycle:
1. **NORMAL**: Baseline legitimate operation ($A(t) = 0$).
2. **ONSET**: Configurable ramp profile (`smoothstep`, `linear`, `exponential`, or `instant`) transitioning $A(t): 0 \to 1$.
3. **ACTIVE**: Full attack intervention ($A(t) = 1$).
4. **TERMINATION**: Configurable ramp down $A(t): 1 \to 0$.
5. **RECOVERY**: System settling back to nominal baseline.

---

## 3. Zero-Leakage Separation Guarantee
Telemetry window tables strictly isolate observable features from ground truth labels:
- **Observable Columns**: QBER, counts, gains, timing histograms, detector imbalance, Fano factor, inter-arrival gaps.
- **Ground Truth Columns**: `ground_truth_attack_active`, `ground_truth_attack_type`, `ground_truth_attack_phase`, `ground_truth_attack_fraction`, `ground_truth_mean_severity`, `ground_truth_attack_id`.
ML models trained on observable columns have **zero access** to ground truth labels during feature extraction.

---

## 4. Multi-Attack Episodes
A single QKD operational run can host multiple non-overlapping attack episodes with independent types, severities, and profiles, verifying model robustness across dynamic operational environments.

*Generated automatically by `scripts/run_attack_validation.py`.*
