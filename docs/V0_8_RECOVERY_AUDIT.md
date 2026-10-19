# QISM V0.8 Comprehensive Repository Recovery & Current-State Audit

**Project:** Quantum Infrastructure Security Monitor (QISM)  
**Milestone:** Version 0.8 — Concept Drift, Distribution Shift, Adaptive Normal Behaviour, Online Monitoring, and Safe Model Adaptation  
**Date:** October 2026  
**Author:** Antigravity Autonomous Research & Software Engineering Agent  

---

## 1. Executive Summary & Audit Context

Prior to introducing Version 0.8 functionality, this audit establishes the empirical state of truth across the QISM repository. Per the V0.8 specification, claims from README files, prior reports, and comments were cross-referenced directly against executable Python source files, configuration schemas, dataset shards, serialized artifacts, and test suites.

### 1.1 Git Working Tree & Repository Health
- **Git Branch:** `main` (synchronized with `origin/main` at commit `869f6cd`: *feat(v0.6): complete QISM simulator and anomaly detection ML benchmarking infrastructure*).
- **Uncommitted Modifications:**
  - `README.md`: Updated with V0.7 documentation and benchmarks.
  - `src/qism/cli.py`: Augmented with commands `visualize`, `v07-benchmark`, `v07-stream`.
  - `src/qism/random.py`: Added seed derivation utilities.
- **Untracked Modules Added in V0.7:**
  - `src/qism/temporal/` (Complete causal temporal pipeline: windowing, feature engine, alarm state machine, models, streaming engine, evaluator, plotter, experiment runner).
  - `src/qism/visualization/` (Visualization suite with 15 figures).
  - `tests/test_v07_*.py` and `tests/test_visualization.py` (7 new test modules, 13 new unit/integration tests).
  - `docs/V0_7_AUDIT.md`, `docs/V0_7_REPORT.md`, `docs/visualization_guide.md`.
  - `outputs/v07/experiments/` (Manifest, summary CSV, and publication plots).
- **Overall Test Suite Status:**
  - **201 passed out of 201 tests** in 210.77 seconds (100% passing).
  - Zero test failures or regressions.
  - V0.7 causal streaming CLI verified: `python -m qism v07-stream` functions with sub-millisecond per-sample latency.

---

## 2. Version-by-Version Inventory of Existing Capabilities

### V0.1 — Core QKD Simulation Foundation
- **Location:** `src/qism/source/`, `src/qism/channel/`, `src/qism/receiver/`, `src/qism/types.py`, `src/qism/units.py`.
- **Implementation Status:** COMPLETE.
- **Physical Capabilities:**
  - Decoy-state BB84 protocol (signal $\mu$, decoy $\nu$, vacuum $\omega$).
  - Weak coherent pulses (WCP) with Poisson photon statistics $P(n) = e^{-\mu}\mu^n / n!$.
  - Polarization encoding in rectilinear ($Z: |H\rangle, |V\rangle$) and diagonal ($X: |D\rangle, |A\rangle$) bases.
  - Fiber channel propagation with Beer-Lambert attenuation $T = 10^{-(\alpha L + L_{\text{sys}})/10}$.
  - Polarization Beam Splitter (PBS) receiver with passive/active basis selection.
  - Ideal and baseline single-photon detectors.

### V0.2 — Physical / Channel / Detector Behaviour
- **Location:** `src/qism/detector/`, `src/qism/channel/`, `src/qism/random.py`.
- **Implementation Status:** COMPLETE.
- **Physical Capabilities:**
  - SPAD avalanche photodiode kinetics.
  - Nonparalyzable dead time ($\tau_{\text{dead}} \approx 50\,\text{ns}$) and count-rate dependent dead time $\tau_{\text{eff}}(\lambda) = \tau_0 + \Delta\tau \frac{\lambda}{\lambda + \lambda_c}$.
  - Prioritized chronological afterpulse queue with multi-trap exponential decay kinetics.
  - Dark count rate modeling with thermal coupling.
  - Birefringent polarization drift and fiber dispersion timing jitter.
  - 5-tier deterministic pseudo-random number generator (PRNG) architecture using SplitMix64 and PCG64.

### V0.3 — Telemetry Infrastructure
- **Location:** `src/qism/telemetry/`, `src/qism/simulation/`.
- **Implementation Status:** COMPLETE.
- **Observable Telemetry (56 pulse columns, 130 aggregated window columns):**
  - Pulse-level events in `events.parquet` partitioned strictly into observable vs ground truth.
  - Windowed telemetry in `telemetry.parquet` (default aggregation window 2,000 pulses).
  - Observables: QBER, sifted key rate, gains ($Q_\mu, Q_\nu, Q_\omega$), detector imbalance $\frac{c_0 - c_1}{c_0 + c_1}$, double-click rates, arrival time moments (variance, skewness, kurtosis), inter-arrival percentiles ($p_{10}, p_{50}, p_{90}$), Fano factor $\sigma^2 / \mu$, autocorrelation coefficients at lags 1 to 5.
  - Physical sensor readings: temperature, humidity, vibration, background stray light.
  - Strict exclusion of simulator ground truth from ML input matrices.

### V0.4 — Attack & Environmental Scenario Modelling
- **Location:** `src/qism/attacks/`, `src/qism/environment/`.
- **Implementation Status:** COMPLETE.
- **Implemented Attacks (6 physical attack types with 5-phase lifecycles):**
  1. `InterceptResendAttack`: Full or probabilistic interception and basis resend.
  2. `ChannelLossAttack`: Physical fiber attenuation and bending.
  3. `SourceManipulationAttack`: Laser pump current perturbation modulating mean photon number $\mu$.
  4. `DetectorManipulationAttack`: Bias voltage or thermal blinding / efficiency reduction.
  5. `TimingAttack`: Chromatic dispersion / arrival time delay modulation.
  6. `PNSAttack`: Photon number splitting on multi-photon pulses.
- **Attack Lifecycle:** `NORMAL -> ONSET -> ACTIVE -> TERMINATION -> RECOVERY`.
- **Dynamic Environmental Couplings:**
  - Continuous-time Ornstein–Uhlenbeck (OU) mean-reverting stochastic processes: $dX_t = \theta (\mu - X_t) dt + \sigma dW_t$.
  - Diurnal sinusoidal cycles: $A \sin(\frac{2\pi t}{P} + \phi)$.
  - Multi-rate physical sensors subject to Gaussian noise, ADC quantization, and packet dropouts.

### V0.5 — Dataset-Generation Framework
- **Location:** `src/qism/datasets/`, `src/qism/population/`, `src/qism/execution/`.
- **Implementation Status:** COMPLETE.
- **Capabilities:**
  - Latin Hypercube Sampling (LHS) across hardware parameter distributions with dependency DAG validation.
  - Parallel batch simulation with `JobScheduler` and `ProcessPoolExecutor`.
  - Atomic shard writing (`manifest.json`, `run.meta.json`, `.complete` markers).
  - Scenarios partitioned into `calibration`, `normal`, `attack`, `generalization`.
  - Zero leakage between train normal and test attack scenarios.

### V0.6 — Baseline Anomaly Detection & ML Benchmarks
- **Location:** `src/qism/ml/`.
- **Implementation Status:** COMPLETE.
- **Models:**
  - Statistical: `MahalanobisDetector`, `RobustCovarianceDetector`.
  - Non-parametric: `IsolationForestDetector`, `OneClassSVMDetector`, `LOFDetector`.
  - Neural: `AutoencoderDetector`, `TemporalAutoencoderDetector`.
- **Preprocessing & Thresholding:**
  - `PreprocessorPipeline` (robust/standard scaling, zero-variance pruning, fit strictly on normal train).
  - `ThresholdCalibrator` (quantile-based on target false alarm rate $\alpha$).
  - Diagnostics: Random feature shuffle, label permutation sanity checks.

### V0.7 — Temporal, Sequential, Streaming & Operational Monitoring
- **Location:** `src/qism/temporal/`.
- **Implementation Status:** COMPLETE & VALIDATED.
- **Temporal Windowing (`windowing.py`):**
  - Causal sliding window builder $W_t = [X_{t-L+1}, \dots, X_t]$ ($L \times d$).
  - Strict trajectory barriers: grouping by `("scenario_id", "run_id")` ensures windows never bridge across scenarios.
  - Labeling strategies: `terminal`, `any`, `majority`, `fraction`.
- **Temporal Feature Engine (`features.py`):**
  - Causal differences ($\Delta x_t, \Delta^2 x_t$).
  - Rolling mean, variance, standard deviation, MAD.
  - Exponentially weighted moving average (EWMA) and causal rate of change.
- **Temporal Alarm State Machine (`alarm.py`):**
  - States: `NORMAL -> SUSPECT -> ALARM -> COOLDOWN`.
  - Dual-threshold hysteresis ($\tau_{\text{high}}, \tau_{\text{low}}$).
  - $K$-of-$N$ persistence filtering (e.g. 2 of 3 anomalous windows required for alarm escalation).
  - Configurable cooldown windows and minimum alarm dwell duration.
- **Causal Temporal Models (`models.py`):**
  - `RollingZScoreDetector`: Rolling normalized Euclidean deviation.
  - `EWMAAnomalyDetector`: EWMA vector deviation with regularized precision matrix.
  - `SequenceAutoencoderDetector`: GRU and LSTM recurrent sequence autoencoders with per-feature error decomposition $E_j$.
- **Causal Online Streaming Inference (`streaming.py`):**
  - `StreamingInferenceEngine` with $O(1)$ ring buffer `deque(maxlen=L)`.
  - Causal left-padding for warmup steps.
  - Zero access to future observations.
  - Sub-millisecond latency profiling ($\sim 0.455\,\text{ms}$ mean latency on CPU).
- **Dual Evaluation Framework (`evaluation.py`):**
  - Sample-level: AUROC, AUPRC, F1, Recall, Precision, FPR.
  - Event-level: Episodic Detection Rate (EDR), Detection Delay distribution ($t_{\text{detect}} - t_{\text{attack}}$), and False Alarms per Simulated Hour.

---

## 3. Analysis of Current Repository State

| Dimension | State | Notes |
|---|---|---|
| Core Simulation | Complete | Decoy-state BB84, SPAD afterpulsing, environmental OU processes. |
| Attack Suite | Complete | 6 physical attacks with 5-phase lifecycles. |
| Datasets | Complete | Datasets in `data/v0.5_small/v05_small_dataset` and `data/v05_experiments/`. |
| Temporal Pipeline (V0.7) | Complete & Passing | 201/201 tests pass; CLI commands active; clean modular structure. |
| Concept Drift (V0.8) | Not Implemented | V0.7 assumes stationary normal distribution. No drift detection, no adaptive references, no safe adaptation state machine, no model versioning or rollback. |
| Redundancy / Duplication | None | Architecture is modular and clean. |

---

## 4. Key Findings Relevant to V0.8

1. **Stationary Threshold Failure Under OOD (Exp 9 in V0.7):**
   - In V0.7, evaluating fixed-threshold models on the OOD generalization dataset showed AUROC dropping from 0.6111 to 0.3125 and FPR jumping from 0.1667 to 0.6250.
   - The detector misidentified legitimate environmental shifts as attacks.
   - This directly motivates the central V0.8 requirement: **adaptive normal behaviour and concept drift tracking.**

2. **Available Physical Parameters for Legitimate Drift:**
   - In `DynamicEnvironment` (`src/qism/environment/dynamic.py`), legitimate physical parameters that drift include:
     - Fiber channel attenuation / optical loss drift (`channel_loss_drift`)
     - Detector temperature drift (`temperature`)
     - SPAD dark counts and efficiency drift (`detector_efficiency_drift`)
     - Polarization alignment drift (`polarization_drift`)
     - Clock frequency and jitter drift (`clock_drift`)
     - Background stray light drift (`background_light`)
     - Source laser pump intensity drift (`source_intensity_drift`)
   - **Critical scientific rule:** V0.8 will utilize *only* these physically simulated parameters for drift generation, avoiding synthetic or arbitrary feature mutations.

3. **Danger of Unconstrained Adaptation:**
   - A naive adaptive detector that recalculates thresholds or reference statistics from incoming telemetry will incorporate attack telemetry during an attack, causing attack sensitivity to collapse.
   - V0.8 must implement **Safe Adaptation Gating**, **Shadow Model Validation**, **Model/Reference Versioning**, and **Rollback**.

---

## 5. Decision & Next Steps for V0.8

With V0.7 validated and all 201 tests passing, we proceed to design and implement V0.8:
- Directory: `src/qism/drift/` (modular, non-monolithic design).
- Submodules:
  - `scenarios.py`: Drift scenario definitions and telemetry generators (abrupt, gradual, recurring, multi-condition, drift + attack).
  - `detectors.py`: Statistical drift detectors (Kolmogorov-Smirnov, Wasserstein distance, Population Stability Index, multivariate aggregated drift).
  - `reference.py`: Reference window management (fixed, sliding, anchored) with contamination safeguards.
  - `adaptation.py`: Safe adaptation gate and state machine (`STABLE -> DRIFT_SUSPECTED -> ADAPTATION_PENDING -> ADAPTING -> VALIDATING -> STABLE_UPDATED / SECURITY_ALERT / ROLLBACK`).
  - `registry.py`: Model and reference versioning registry with rollback capabilities.
  - `streaming.py`: Adaptive streaming inference engine extending V0.7 streaming architecture.
  - `evaluation.py`: V0.8 metrics (drift detection delay, adaptation acceptance/rejection, attack recall retention).
  - `experiments.py`: Benchmark runner orchestrating Experiments 1–12 (fixed vs naive vs safe adaptation).
  - `plots.py`: V0.8 visualization suite (drift scores, feature heatmaps, distribution comparison, version timeline).
- Outputs directory: `outputs/v08/`.
- Test suite: `tests/test_v08_*.py`.
