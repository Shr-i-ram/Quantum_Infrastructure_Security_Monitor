# QISM V0.7 Comprehensive Repository and Recovery Audit

**Project:** Quantum Infrastructure Security Monitor (QISM)  
**Target Milestone:** Version 0.7 — Temporal, Sequential & Streaming Anomaly Detection  
**Audit Date:** October 2026  
**Auditor:** Antigravity AI Autonomous Research Engineer  

---

## 1. Executive Summary & Recovery State Assessment

This document establishes the authoritative pre-implementation audit for Version 0.7 of QISM, strictly adhering to the requirements of the project specification.

### 1.1 Recovery State Determination
A complete audit of the repository state reveals:
- **Baseline Version:** The codebase is at **Version 0.6.0** (commit `869f6cd`: *feat(v0.6): complete QISM simulator and anomaly detection ML benchmarking infrastructure*).
- **Recent Uncommitted Modifications:** The working directory contains additions for the visualization presentation suite (`src/qism/visualization/`, `scripts/generate_visualizations.py`, `tests/test_visualization.py`, `docs/visualization_guide.md`, and CLI command `visualize` in `src/qism/cli.py`).
- **V0.7 Code State:** **No V0.7 implementation files, scripts, or tests exist yet.** The previous agent completed V0.6 and the visualization subsystem, but did not implement V0.7 components.
- **Test Suite Status:** The existing test suite was executed in full: **188 passed out of 188 tests** in 222.54 seconds. The underlying simulator (V0.1–V0.4), dataset infrastructure (V0.5), ML baseline layer (V0.6), and visualization suite are functionally sound.

---

## 2. Inventory of Existing Codebase Modules

| Module / Path | Version Scope | Implementation Status | Core Responsibilities |
|---|---|---|---|
| `src/qism/config.py` | V0.1–V0.5 | COMPLETE | Pydantic v2 configuration schema for physics, source, channel, detectors, environment, attacks, telemetry, population, and dataset splits. |
| `src/qism/random.py` | V0.1–V0.5 | COMPLETE | 5-tier deterministic seed derivation hierarchy using `splitmix64` and PCG64. |
| `src/qism/types.py` | V0.1–V0.4 | COMPLETE | Enums for `Basis`, `IntensityClass`, `ClickType`, `DetectorState`, etc. |
| `src/qism/source/` | V0.1–V0.2 | COMPLETE | Decoy-state BB84 pulse generation (signal, decoy, vacuum) with Poisson photon statistics. |
| `src/qism/channel/` | V0.1–V0.4 | COMPLETE | Beer-Lambert optical attenuation, polarization misalignment drift, timing jitter. |
| `src/qism/receiver/` | V0.1–V0.2 | COMPLETE | Polarization beam splitter (PBS) routing and passive/active basis selection. |
| `src/qism/detector/` | V0.2–V0.4 | COMPLETE | SPAD avalanche kinetics, nonparalyzable & rate-dependent dead time, afterpulse priority queue, dark counts. |
| `src/qism/environment/` | V0.4 | COMPLETE | Continuous Ornstein–Uhlenbeck drift processes, diurnal cycles, physical couplings, multi-rate physical sensors. |
| `src/qism/attacks/` | V0.3–V0.4 | COMPLETE | 6 physical attacks (`InterceptResend`, `ChannelLoss`, `SourceManipulation`, `DetectorManipulation`, `Timing`, `PNS`) with 5-phase lifecycle (`NORMAL`, `ONSET`, `ACTIVE`, `TERMINATION`, `RECOVERY`). |
| `src/qism/telemetry/` | V0.1–V0.4 | COMPLETE | Pulse-level events schema (56 columns) and window aggregation (`telemetry.parquet`, 130 columns) with strict observable vs ground-truth categorization. |
| `src/qism/simulation/` | V0.1–V0.4 | COMPLETE | Chunked `Simulator`, atomic `CheckpointManager`, and `RunManager`. |
| `src/qism/datasets/` | V0.1–V0.5 | COMPLETE | Dataset shard writers, manifests, atomic completion markers, and calibration/normal/attack generators. |
| `src/qism/population/` | V0.5 | COMPLETE | LHS parameter distributions, dependency DAG resolution, stress regimes. |
| `src/qism/execution/` | V0.5 | COMPLETE | Parallel `JobScheduler` and `ProcessPoolExecutor` with deterministic worker seeds. |
| `src/qism/ml/data/` | V0.6 | COMPLETE (NEEDS REVISION) | Scenario-level partitioning (`SplitManager`) and Parquet loading (`MLDatasetLoader`). |
| `src/qism/ml/features/` | V0.6 | COMPLETE (NEEDS REVISION) | Observable feature extraction (`FeatureExtractor`) and predefined ablation feature sets. |
| `src/qism/ml/preprocessing/`| V0.6 | COMPLETE | `PreprocessorPipeline` fit strictly on train normal with JSON serialization. |
| `src/qism/ml/models/` | V0.6 | COMPLETE (NEEDS EXTENSION)| 7 anomaly detector baselines (`Mahalanobis`, `RobustCovariance`, `IsolationForest`, `OneClassSVM`, `LOF`, `Autoencoder`, `TemporalAutoencoder`). |
| `src/qism/ml/thresholding/` | V0.6 | COMPLETE | `ThresholdCalibrator` (quantile-based) and `AlarmStateMachine` (3 states, persistence $K$, hysteresis). |
| `src/qism/ml/evaluation/` | V0.6 | COMPLETE | Window-level classification metrics, continuous ROC/PR curves, and `AttackInstanceEvaluator` (delay, false alarms). |
| `src/qism/ml/experiments/` | V0.6 | COMPLETE | `ExperimentConfig`, `ExperimentRunner`, and leakage sanity checks (label shuffle, random features). |
| `src/qism/visualization/` | V0.6 | COMPLETE | Publication-grade 15 presentation figures suite. |

---

## 3. Detailed Audit of Versions 0.1 Through 0.5

### 3.1 Physical and Event Foundation (V0.1–V0.4)
- **Protocol & Source:** Decoy-state BB84 operating with weak coherent laser pulses. Each pulse emits $n \sim \text{Poisson}(\mu)$ photons with randomized polarization states in either rectilinear ($Z$) or diagonal ($X$) bases.
- **Channel:** Transmittance governed by Beer-Lambert law $T = 10^{-(\alpha L + L_{\text{sys}})/10}$. Polarization undergoes stochastic drift; arrival times include fiber chromatic dispersion and propagation jitter.
- **Detectors:** Two-detector Bob ($D_0, D_1$). Realistic SPAD kinetics: nonparalyzable dead time ($\tau_{\text{dead}} \approx 50\,\text{ns}$), rate-dependent dead time $\tau_{\text{eff}}(\lambda) = \tau_0 + \Delta\tau \frac{\lambda}{\lambda + \lambda_c}$, exponential afterpulsing modeled via a prioritized chronological queue, and temperature-dependent dark counts.
- **Coupled Environmental Dynamics:** Continuous-time Ornstein–Uhlenbeck (OU) mean-reverting processes and diurnal cycles dynamically modulate SPAD temperature, dark counts, and channel attenuation. Multi-rate physical sensors record ambient observables subject to Gaussian noise, ADC quantization, and packet dropouts.
- **Physical Attack Engine:** Attacks do not manipulate high-level telemetry features; they intervene directly in physical simulation parameters (e.g. Eve intercepts pulses and resends new quantum states, optical attenuation is added, laser diode pump current is perturbed, SPAD bias voltage is modulated, or arrival timing is delayed).
- **Event-Level Telemetry Schema:** 56 columns in `events.parquet` partitioned into explicit `FieldCategory` classes (`OBSERVABLE_TELEMETRY`, `DERIVED_TELEMETRY`, `SIMULATOR_GROUND_TRUTH`, `FUTURE_ONLY`).

### 3.2 Dataset Architecture (V0.5)
- **Scenario vs Run:** A `Scenario` defines frozen hardware parameters and environment conditions ($d_{\text{km}}, \eta_0, \eta_1, T_{\text{baseline}}, \text{attack}$); a `Run` represents an independent stochastic Monte Carlo realization.
- **Windowed Telemetry Parquet:** Events are aggregated by pulse ID into fixed windows (default 2,000 pulses). `telemetry.parquet` contains 130 columns:
  - 40+ observable features: QBER, sifted rate, gains (signal, decoy, vacuum), detector imbalance, double-click rate, timing moments, inter-arrival percentiles, Fano factor, autocorrelation lags 1..5, sensor readings.
  - Sidecar ground-truth labels: `ground_truth_attack_active`, `ground_truth_attack_type`, `ground_truth_attack_phase`, `ground_truth_attack_fraction`, `ground_truth_mean_severity`, `ground_truth_attack_id`.
- **Temporal Consistency:** Within each scenario and run, `window_id` is strictly monotonic and contiguous ($0, 1, 2, \dots$), corresponding to continuous time intervals $t \in [w \cdot T_{\text{win}}, (w+1) \cdot T_{\text{win}})$.

---

## 4. Rigorous Scientific & Data Leakage Audit of Version 0.6

An in-depth code and methodological audit of the V0.6 implementation was performed to identify any methodological flaws or leakage vectors.

### 4.1 Positive Scientific Strengths of V0.6
1. **Zero Feature-Level Leakage:** `FeatureExtractor` strictly rejects any column in `GROUND_TRUTH_FIELDS` or with prefix `ground_truth_`, `true_`, or `yield_`. Automated unit tests (`tests/test_v06_feature_leakage.py`) enforce this.
2. **Sanity Diagnostics:** Automated label shuffling and random feature tests confirm that models trained on legitimate telemetry collapse to random guessing ($\text{AUROC} \approx 0.50$) when labels or features are destroyed.
3. **Normal-Only Training Paradigm:** Models are fit exclusively on legitimate normal operational data, respecting the unsupervised one-class monitoring principle.
4. **Normal-Only Thresholding:** Thresholds are calibrated on validation scores using target false positive rate quantiles $\tau = Q_{1-\alpha}(S_{\text{val\_normal}})$.

### 4.2 Critical Methodological Flaws Identified in V0.6

#### Flaw 1: Scenario Boundary Leakage in Temporal Windows
- **Location:** `src/qism/ml/models/temporal.py`, method `TemporalAutoencoderDetector._create_sequences(X_2d)`.
- **Issue:** The sliding sequence builder converts a 2D array of shape $(N, d)$ into 3D sequences $(N, L, d)$ using a simple loop `sub = X_2d[max(0, i - L + 1) : i + 1]`. When `X_2d` represents a concatenation of multiple scenarios (as in `df_train` or `df_test`), **sliding windows cross scenario and run boundaries**. At the start of Scenario $B$, the sequence contains the trailing windows of Scenario $A$.
- **Impact:** In the test set, an attack scenario's onset window contains normal observations from the preceding scenario, or observations from a completely different attack type.
- **Required V0.7 Fix:** Sequences must be constructed strictly within bounded runs/scenarios. A sequence must **never** bridge across scenario boundaries.

#### Flaw 2: Cross-Scenario Grouping Collision in Feature Extractor
- **Location:** `src/qism/ml/features/extractor.py`, lines 79–86:
  ```python
  group_col = df["run_id"] if "run_id" in df.columns else None
  rm = col_series.groupby(group_col).rolling(window=L, min_periods=1).mean()
  ```
- **Issue:** In the V0.5 dataset structure, every scenario contains runs indexed locally as `"run_0000"`. When `df` contains multiple scenarios, grouping by `run_id` alone groups together `"run_0000"` from `normal_0000`, `normal_0001`, and `attack_0000` into a single group!
- **Impact:** Rolling statistics mix values across completely different physical deployments.
- **Required V0.7 Fix:** Grouping must be performed on the unique tuple `("scenario_id", "run_id")`.

#### Flaw 3: Validation Set Contamination in Test Set Evaluation
- **Location:** `src/qism/ml/data/splits.py`, lines 67–72:
  ```python
  partitions = {
      "train": train_normal,
      "val": val_normal if val_normal else train_normal,
      "test": by_split["attack"] + val_normal,  # <-- val_normal is placed in test
      "generalization": by_split["generalization"],
  }
  ```
- **Issue:** In `runner.py`, `val` is used to calibrate the threshold $\tau$. Then `test` evaluates metrics on `attack + val_normal`. Evaluating the false-alarm rate and specificity on the exact same normal scenarios used for threshold calibration means test-set normal metrics are not out-of-sample.
- **Required V0.7 Fix:** When sufficient normal scenarios exist, allocate an independent `test_normal` partition disjoint from `val_normal`.

#### Flaw 4: Total Absence of Streaming & Real-Time Causal Architecture
- **Issue:** `TemporalAutoencoderDetector` only accepts a static 2D batch array, creates sequences over the entire array, and performs a single PyTorch forward pass. There is no stateful streaming engine (`update(sample) -> score, alarm, state`), no buffering, and no latency benchmarking.
- **Required V0.7 Fix:** Implement a formal causal streaming inference subsystem.

#### Flaw 5: Insufficient Temporal Trajectory Depth in Small Dataset
- **Issue:** In `data/v0.5_small/v05_small_dataset`, runs consist of 10,000 pulses with window size 2,000 pulses, producing only **5 windows per run**. In a 5-window sequence, setting sequence length $L = 3$ means windows 0 and 1 are artificial edge padding, window 2 is the first full window, and windows 3–4 are the end of the run. This is insufficient to study attack onset dynamics, detection delay, persistence filtering, and gradual degradation.
- **Required V0.7 Fix:** Generate or configure a dataset with sufficient temporal depth (e.g. 50–100+ windows per trajectory) to properly evaluate temporal dynamics, while retaining support for the smoke/small datasets.

---

## 5. Architectural Blueprint for Version 0.7

To elevate QISM from point-wise anomaly classification to temporal, sequential, and streaming operational monitoring, Version 0.7 will implement the following components:

```
+----------------------------------------------------------------------------------------------------+
|                                    QISM V0.7 TEMPORAL ARCHITECTURE                                 |
+----------------------------------------------------------------------------------------------------+

       Raw Telemetry Stream (Window-level X_t, strictly causal)
                       │
                       ▼
       ┌────────────────────────────────────────────────────────┐
       │             BOUNDARY-AWARE TEMPORAL WINDOWING          │
       │  - Group by (scenario_id, run_id)                      │
       │  - Sliding Window: W_t = [X_{t-L+1}, ..., X_t]         │
       │  - Configurable Window Length L, Stride S              │
       │  - Explicit Metadata Sidecar (run, scenario, t_start)  │
       │  - Multi-Strategy Labeling: Terminal, Any, Majority, r │
       └──────────────────────────┬─────────────────────────────┘
                                  │
       ┌──────────────────────────┴─────────────────────────────┐
       ▼                                                        ▼
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│     TEMPORAL FEATURE EXTRACTOR       │     │     SEQUENCE ENCODER-DECODER         │
│  - First & Second Differences (d, d2)│     │  - Causal GRU / LSTM Sequence AE     │
│  - Rolling Mean, Std, Variance       │     │  - Latent Representation z           │
│  - Causal EWMA & Rate of Change      │     │  - Reconstruction Error: E(W)        │
│  - Cross-stream physics correlations │     │  - Feature Decomposition: E_j        │
└──────────────────┬───────────────────┘     └──────────────────┬───────────────────┘
                   │                                            │
                   ▼                                            ▼
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│     TEMPORAL STATISTICAL BASELINES   │     │      STREAMING INFERENCE ENGINE      │
│  - Rolling Z-Score / Mahalanobis     │     │  - Causal update(sample) API         │
│  - Rolling Median / MAD              │     │  - Ring buffer history (length L)    │
│  - Classical ML with Temporal Feats  │     │  - Per-sample computational latency  │
└──────────────────┬───────────────────┘     └──────────────────┬───────────────────┘
                   │                                            │
                   └─────────────────────┬──────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │       TEMPORAL ALARM STATE MACHINE        │
                   │  - States: NORMAL -> SUSPECT -> ALARM     │
                   │  - Persistence: K-of-N consecutive        │
                   │  - Hysteresis: tau_high activation,       │
                   │                tau_low recovery           │
                   │  - Cooldown & Min Alarm Duration          │
                   └─────────────────────┬─────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│       SAMPLE-LEVEL EVALUATION        │     │        EVENT-LEVEL EVALUATION        │
│  - Precision, Recall, F1             │     │  - Episode Detection Rate (Recall)   │
│  - FPR, Specificity                  │     │  - False Alarm Episodes & Per Hour   │
│  - Continuous AUROC, AUPRC           │     │  - Detection Delay: D = t_det - t_atk│
│  - Score Separation Ratio            │     │  - Alarm Duration & Persistence      │
└──────────────────────────────────────┘     └──────────────────────────────────────┘
```

---

## 6. Action Plan & Remaining Work Checklist

1. [x] **Repository Audit Complete:** Document current architecture, test status, and V0.6 findings in `docs/V0_7_AUDIT.md`.
2. [ ] **Module 1: `qism.temporal.windowing`:** Implement boundary-aware sliding window builder, metadata preservation, and 4 labeling strategies (Terminal, Any, Majority, Fraction).
3. [ ] **Module 2: `qism.temporal.features`:** Implement causal temporal feature engineering (first/second differences, rolling mean/variance/std/MAD, EWMA, rate-of-change, correlation proxies) with strict `(scenario_id, run_id)` isolation.
4. [ ] **Module 3: `qism.temporal.alarm`:** Implement enhanced operational alarm state machine supporting K-of-N persistence, hysteresis, cooldown, and minimum alarm duration.
5. [ ] **Module 4: `qism.temporal.streaming`:** Implement stateful `StreamingDetector` abstraction with causal ring buffer, `update(sample)` interface, and computational latency profiling.
6. [ ] **Module 5: `qism.temporal.models`:** Implement sequence-aware reconstruction models (boundary-aware GRU/LSTM Autoencoder) with feature-level score decomposition ($E_j$).
7. [ ] **Module 6: `qism.temporal.evaluation`:** Implement dual sample-level and event-level evaluation, detection delay ($D = t_d - t_a$), and operational false alarms per hour.
8. [ ] **Module 7: Experiment Matrix & Runner:** Implement reproducible experiment runner for Experiments 1 through 10 (point baseline, temporal stats, persistence ablation, sequence autoencoder, window-length study, streaming latency, OOD generalization).
9. [ ] **Module 8: Visualizations:** Implement temporal telemetry timelines, anomaly scores, raw vs filtered alarms, detection delay distributions, score decomposition heatmaps, and window-length comparison curves.
10. [ ] **Module 9: CLI Integration:** Expose unified commands in `qism.cli` (`v07-train`, `v07-evaluate`, `v07-stream`, `v07-benchmark`, `v07-experiment`).
11. [ ] **Module 10: Testing & Validation:** Full unit tests for windowing, features, streaming causality, alarm logic, event metrics, and leakage checks.
12. [ ] **Module 11: Comprehensive Report & Documentation:** Write `docs/V0_7_REPORT.md` and update `README.md`.
