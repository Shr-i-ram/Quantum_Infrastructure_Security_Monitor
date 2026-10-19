# QISM V1.0 Security & Information-Flow Audit Report

**Project:** Quantum Infrastructure Security Monitor (QISM)  
**Milestone:** V1.0 — Integrated, Validated, Reproducible Research Prototype  
**Date:** October 2026  
**Auditor:** Antigravity Autonomous Research & Software Engineering Agent  

---

## 1. Executive Summary & Audit Scope

This security and information-flow audit evaluates the architectural safeguards, causality guarantees, and anti-contamination boundaries of the Quantum Infrastructure Security Monitor (QISM) V1.0 prototype.

The audit evaluates six core security pillars:
1. **Information-Flow Boundaries:** Clear separation between physical simulation, observable telemetry, operational inference, and offline evaluation sidecars.
2. **Ground-Truth & Metadata Leakage:** Strict prevention of labels (`attack_type`, `attack_active`, `is_attack`, `ground_truth_*`, `scenario_id`) entering feature matrices or decision pipelines.
3. **Temporal Causality:** Absolute enforcement of unidirectional time flow ($t_0 \le t_1 \le \dots \le t_{\text{current}}$), causal sliding window buffers, and absence of future lookahead or future-derived normalizers.
4. **Adaptation Safety & Reference Integrity:** Formal evaluation of candidate adaptation gating during anomalous episodes, preventing adversary-poisoned reference models.
5. **Operational State Integrity & Fail-Safe Behaviour:** Handling of malformed, non-numeric, or missing telemetry without generating false nominal security assessments.
6. **System Boundaries & Non-Execution Policy:** Enforcement that all operator recommendations remain strictly advisory (`is_automated_execution = False`), with zero automated actuation on physical optical components or key material.

---

## 2. Information-Flow Boundaries

QISM enforces a strict one-way information pipeline:

```
[Simulation / Physics Core]
           │
           ▼
[Observable Telemetry (X_t)]  ──(Sidecar Isolation)──► [Ground Truth & Provenance (y_t)]
           │                                            (Accessible ONLY to Offline Evaluation)
           ▼
[Telemetry Ingestion & Sanity]
           │ (Rejects NaNs, Non-Monotonic Timestamps, Sequence Duplicates)
           ▼
[Causal Temporal Buffer (W_t)]
           │
     ┌─────┴─────┐
     ▼           ▼
 [Anomaly]   [Drift]
     │           │
     └─────┬─────┘
           ▼
[Multivariate Signal Fusion]
           │
           ▼
[Operational Alarm Machine]
           │
           ▼
[Incident Lifecycle Engine]
           │
           ▼
[Deterministic Severity Engine]
           │
           ▼
[Structured Alerts & Audit Log] ──(Operator Notifications / Advisory Actions ONLY)
```

### Verification Evidence:
- Observable feature extraction in `src/qism/telemetry/features.py` derives metrics solely from event counts, arrival histograms, and photon click statistics. Ground-truth simulator parameters (e.g. `afterpulse_rate_ground_truth`, `true_loss_delta_db`) are explicitly omitted from observable matrices.
- `src/qism/monitor/ingestion.py` scans all ingested records against `FORBIDDEN_GROUND_TRUTH_KEYS`. Any matching fields are routed exclusively to `record.metadata` (for post-hoc evaluation) and completely stripped from `record.features`.

---

## 3. Ground-Truth & Feature Leakage Audit

### 3.1 Tested Leakage Vectors
1. **Attack Identification Features:** Fields such as `attack_active`, `attack_type`, `attack_strength`, `is_attack` must never be accessible to the anomaly detector, drift detector, signal fusion, or incident manager.
2. **Scenario and Run IDs:** Provenance identifiers (`scenario_id`, `run_id`, `session_id`) must never be used as predictive inputs.
3. **Normalizer Leakage:** Normalization parameters (mean, standard deviation) must be fit exclusively on unperturbed calibration baseline data, never on evaluation or test trajectories.
4. **Threshold Selection Leakage:** Anomaly thresholds ($\tau_{\text{anom}}$) and drift thresholds ($\tau_{\text{drift}}$) must be calibrated on normal operational validation sets, never optimized post-hoc against attack test labels.

### 3.2 Audit Findings
- **Feature Definitions Check:** In `tests/test_v06_feature_leakage.py`, `is_ground_truth_field()` was executed against all 16 extracted observable features. 0 ground-truth fields were present in feature matrices.
- **Ingestion Boundary Check:** In `tests/test_v09_ingestion.py`, telemetry records containing injected `attack_type="intercept_resend"` and `attack_active=True` were ingested. The resulting `TelemetryRecord.features` dictionary contained only valid physical observables.
- **Evaluation Isolation:** The evaluation layer (`src/qism/monitor/evaluation.py`) receives `MonitorRunResult` alongside ground-truth labels purely to calculate event-level recall, detection latency, and false positive rates post-hoc. The operational orchestrator (`QISMMonitor`) has no reference to ground truth.

---

## 4. Temporal Causality Audit

### 4.1 Causal Windowing Invariants
A causal monitor at step $t$ must depend strictly on observations $\{X_\tau\}_{\tau \le t}$:
1. **Sliding Buffer Construction:** `qism.temporal.windowing` constructs windows $W_t = [X_{t-L+1}, \dots, X_t]$. Future samples $X_{t+k}$ ($k > 0$) are structurally inaccessible.
2. **Monotonicity Enforcement:** `TelemetryIngestor` checks that $t_{\text{current}} \ge t_{\text{previous}}$. If a clock jump backwards or out-of-order packet arrives, a `TelemetryValidationError` is raised, preventing temporal contamination.
3. **Equivalence of Batch and Streaming Ingestion:** `tests/test_v09_orchestrator_replay.py` confirms that feeding a dataset as a single DataFrame versus yielding records sequentially in real-time streaming produces identical numerical scores, incident boundaries, and audit log entries.

### 4.2 Causality Audit Verdict
**Passed.** The operational pipeline is strictly causal with zero future lookahead.

---

## 5. Adaptation Safety & Anti-Contamination Invariants

### 5.1 Threat Model
In an online adaptive monitoring architecture, an adversary executing a stealthy or gradual attack (e.g. slowly increasing eavesdropping intercept rate or gradual detector blinding) seeks to "poison" the baseline distribution:
$$\text{If } X_t^{\text{attack}} \text{ is assimilated into reference } \mathcal{R}, \text{ the attack is normalized and future attacks go undetected.}$$

### 5.2 Anti-Contamination Gate Implementation
In `src/qism/drift/adaptation.py` and `src/qism/monitor/orchestrator.py`:
1. **Anomaly-Gated Candidate Collection:** When drift is detected, candidate telemetry samples are buffered for shadow validation. However, if the rolling anomaly score exceeds the safety threshold:
   $$A_t \ge \tau_{\text{safe}} = 0.85 \cdot \tau_{\text{anom}}$$
   or if any temporal alarm is active, candidate collection is immediately aborted (`AdaptationDecision.REJECTED_ANOMALY`).
2. **Incident-Gated Lockout:** Active incidents in `QISMMonitor` actively inhibit reference adaptation.
3. **Shadow Validation:** Before any candidate reference is promoted, it is benchmarked against a stationary calibration holdout.
4. **Auditability:** Every adaptation step (`ADAPTATION_STARTED`, `MODEL_PROMOTED`, `ADAPTATION_REJECTED`) is recorded in the append-only JSONL audit log with active model and reference version IDs.

### 5.3 Audit Findings & Test Results
- In `tests/test_v08_security.py` and Benchmark Scenario 6 (*Attack During Drift*), attack telemetry was injected during an active drift transition.
- **Observed Result:** In 100% of evaluated attack injection runs, the safety gate triggered, candidate promotion was rejected, and the baseline reference remained pristine.

---

## 6. Operational State Integrity & Fail-Safe Invariants

### 6.1 Handling Malformed Inputs
- **NaN / Infinity Detection:** `TelemetryIngestor` rejects non-finite values (`math.isnan(v)` or `math.isinf(v)`) immediately.
- **Physical Bounds Violation:** Telemetry exceeding physical realities (e.g. QBER $> 1.0$, negative click rates) raises `TelemetryValidationError`.
- **Fail-Safe Principle:** Ingestion failures increment `total_rejected` and log errors. The pipeline never defaults a corrupt sample to a nominal "0.0 anomaly score", preventing false negatives caused by data corruption.

### 6.2 Checkpointing & Recovery Integrity
- The orchestrator serializes complete internal state: step counter, rolling buffers, alarm state machine status, incident manager records, and version hashes.
- In `tests/test_v09_orchestrator_replay.py`, interrupting a run at step 60, serializing to JSON, restoring to a clean orchestrator, and resuming produced bit-for-bit identical results to uninterrupted execution.

---

## 7. Operational Boundaries & Non-Execution Policy

### 7.1 Advisory-Only Response Engine
The severity engine (`src/qism/monitor/severity.py`) generates structured response actions (e.g. `INVESTIGATE_OPTICAL_FIBER`, `SWITCH_QUANTUM_CHANNEL`, `PAUSE_KEY_GENERATION`).
- Every recommendation object explicitly sets:
  ```python
  is_automated_execution = False
  ```
- **Physical Actuation Guarantee:** QISM contains zero code, network sockets, or hardware drivers capable of actuating optical attenuators, bias voltages, optical switches, or key deletion routines. All outputs are strictly advisory operator guidance.

---

## 8. Summary of Findings & Remaining Risks

| Category | Requirement | Audit Result | Notes |
|---|---|---|---|
| **Ground-Truth Leakage** | Forbidden labels absent from features | **PASSED (Zero Leakage)** | Enforced in `ingestion.py` and `features.py` |
| **Temporal Causality** | No future information used | **PASSED (Strictly Causal)** | Validated by streaming equivalence tests |
| **Adaptation Safety** | Attacks prevented from poisoning reference | **PASSED (100% Rejection)** | Gated at $A_t \ge 0.85 \tau_{\text{anom}}$ and active alarm |
| **Fail-Safe Ingestion** | Malformed inputs handled safely | **PASSED (Explicit Validation)** | Explicit `TelemetryValidationError` raised |
| **State Recovery** | Resumption matches uninterrupted run | **PASSED (Exact Match)** | Validated in Exp 11 and unit tests |
| **Actuation Boundary** | No automated physical countermeasures | **PASSED (Advisory Only)** | `is_automated_execution = False` enforced |

### Remaining Limitations & Considerations:
1. **Simulation vs Hardware Realities:** The current telemetry is generated by numerical simulation of decoy-state BB84. Physical hardware testbeds may introduce unmodelled electromagnetic interference or non-stationary fiber noise requiring re-tuning of $\tau_{\text{anom}}$ and $\tau_{\text{drift}}$.
2. **Threshold Sensitivity:** In environments with heavy non-adversarial ambient variations, thresholds calibrated on small calibration sets may experience elevated false-alarm rates unless safe adaptation is enabled.
