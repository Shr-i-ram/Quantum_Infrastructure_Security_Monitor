# QISM V1.0 Benchmark Summary Report

**Release Artifact:** Quantum Infrastructure Security Monitor (QISM) V1.0  
**Generated:** October 2026  
**Status:** Validated Research Prototype  

## 1. Multi-Seed Benchmark Summary (36 Runs across Seeds 42, 101, 2024)

- **Total Scenarios Evaluated:** 12
- **Seeds Evaluated:** 42, 101, 2024
- **Attack Incident Recall:** 100% across all evaluated attack scenarios (Scenarios 04, 05, 06, 07, 08, 09, 12).
- **Incident Segmentation (Intermittent Attack, Scenario 09):** 100% recall and 100% precision cleanly separating distinct attack bursts.
- **Alert Compression Ratio:** Up to **92.7x** under sustained attack conditions (Scenario 05), achieving a **95.7% alert volume reduction**.
- **Average Processing Latency:** $6.04\text{ ms}$ to $6.87\text{ ms}$ per step ($141.9$ to $178.3\text{ steps/s}$).

## 2. Required Ablation Summary

1. **Comparison A (Progression):** False alarms during benign loss drift reduced from 45 (V0.6) and 38 (V0.7) down to 0 (V1.0) while retaining 100% attack detection.
2. **Comparison B (Fixed vs Adaptive):** Fixed monitoring incurred persistent false alarms under +2.5 dB channel drift; safe adaptive monitoring eliminated false alarms while freezing during attacks.
3. **Comparison C (Raw Alarms vs Incidents):** 93 raw alarms correlated into 1 incident and 14 lifecycle notifications (93.0x compression, 84.9% alert reduction).
4. **Comparison D (Single Signal vs Fusion):** Fused evidence achieved Incident F1 = 1.00 and 93.0x compression, preventing the incident fragmentation seen with single anomaly scores (F1 = 0.50).
5. **Comparison E (Performance):** Mean latency 6.60 ms, P95 latency 10.65 ms, throughput 147.4 steps/s.

## 3. Robustness & Integrity
- **Malformed Input Stress (Scenario 11):** 100% rejection of NaNs, out-of-order timestamps, and physical violations with explicit `TelemetryValidationError` and high-severity audit logging.
- **Streaming Checkpoint Recovery (Scenario 12):** Bit-exact state resumption verified across all 3 seeds.
- **Ground-Truth Isolation:** Verified zero leakage of simulation sidecars into operational features.
