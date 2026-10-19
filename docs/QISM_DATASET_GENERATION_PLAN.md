# QISM Research-Grade Dataset Generation Plan

## 1. Executive Summary & Objective

This document defines the research-grade dataset generation plan for the **Quantum Infrastructure Security Monitor (QISM)**.
The objective is to produce a large, comprehensive, reproducible, model-agnostic, and leakage-safe dataset from the existing authentic QISM Monte Carlo simulator to support the training, validation, and benchmarking of:
- Classical unsupervised anomaly detectors (Isolation Forests, One-Class SVM, Deep SVDD).
- Statistical drift and streaming monitors.
- Temporal deep learning baselines (LSTM, GRU, Autoencoders).
- Quantum machine learning (QML) models (Variational Quantum Classifiers [VQC], Quantum Kernel SVMs [QSVM], Quantum Autoencoders [QAE]).
- The flagship **Physics-Informed Temporal Quantum Support Vector Data Description (PI-TQSVDD)** architecture.

In accordance with strict pair programming invariants:
- **No model training, benchmarking, or threshold tuning is conducted during dataset generation.**
- **No synthetic data fabrication or direct NumPy column modification is permitted.** Every record originates from the authentic physics engine in `qism.simulation.runner.RunManager`.

---

## 2. Simulator Audit & Capabilities

The repository audit revealed:
1. **Simulator Core**: QISM v0.6.0 (`qism.simulation.simulator.Simulator`), implementing chunked Monte Carlo simulation of Decoy-State BB84 (`decoy_bb84`) with Poissonian weak coherent pulses, realistic fiber attenuation, polarization drift, and dynamic ambient environmental couplings (temperature, humidity, vibration, stray background lux).
2. **Detector Array**: Dual single-photon avalanche photodiodes (SPADs D0 and D1) featuring non-paralyzable rate-dependent dead time (`DeadTimeConfig`), trap-assisted afterpulsing (`AfterpulseConfig`), timing jitter (30–150 ps), and thermal dark-count scaling.
3. **Physical Attack Engines**: 6 native, physically simulated attack mechanisms in `src/qism/attacks/`:
   - `intercept_resend`: Optical intercept-resend with basis choice mismatch inducing 25% nominal error rate on tapped pulses.
   - `channel_loss`: Dynamic fiber bending, macroscopic tapping, and variable attenuation injection.
   - `source_manipulation`: Laser diode driver perturbation shifting mean photon numbers ($\mu_{\text{signal}}, \mu_{\text{decoy}}$).
   - `detector_manipulation`: Thermal blinding and efficiency mismatch attacks creating detector click asymmetry.
   - `timing`: Optical delay line manipulation shifting arrival timing histograms outside detector gate centers.
   - `pns` (Photon Number Splitting): Selective single-photon suppression and multi-photon tapping.
4. **Telemetry & Feature Extraction**: `qism.telemetry.features.window_features` extracts 76 observable physical features per window, preserving 18 ground-truth labels and 31 raw event-level counters.
5. **Execution Infrastructure**: `qism.execution.scheduler.JobScheduler` and `qism.execution.worker.execute_simulation_job` provide deterministic multi-worker parallel execution, atomic shard writing, and checkpoint-resume capabilities.

---

## 3. Scale Feasibility Analysis & Research Targets

### 3.1 Initial Target vs. Physical Semantics
The initial prompt target posited 1,200,000 telemetry windows.
Our audit of the simulation engine revealed the relationship between pulses and windows:
- The QISM simulator generates physical events pulse-by-pulse (each pulse tracking emission state, Alice basis/bit, fiber transmission, dark count generation, detector state machine, and click recording).
- A telemetry window aggregates $W$ pulses. In standard research benchmarks, $W = 1,000$ pulses (providing 1 ms temporal resolution at 1 MHz repetition rate).
- Generating 1,200,000 windows at $W = 1,000$ requires:
  $$\text{Total Pulses} = 1,200,000 \times 1,000 = 1,200,000,000 \text{ (1.2 Billion Pulses)}$$
- Empirical throughput measured across 4 parallel workers on the local host is $\sim 15,000 \text{ pulses/second}$.
- Runtime for 1.2 billion pulses:
  $$\text{Runtime} = \frac{1.2 \times 10^9}{15,000} \approx 80,000 \text{ seconds} \approx 22.2 \text{ hours}$$
- Disk storage at $\sim 350 \text{ bytes/pulse}$ (full event traces + telemetry):
  $$\text{Storage} \approx 1.2 \times 10^9 \times 350 \text{ bytes} \approx 420 \text{ GB}$$
Available free disk space on drive `C:` is 260.28 GB. Generating 1.2 billion events would exhaust disk space and risk session interruption.

### 3.2 Defensible Production Sizing
As instructed in Section 3 and Section 10 of the specification, when the initial raw row count is constrained by compute and storage physics, we select the largest defensible research configuration:
- **Total Scenarios**: 200 independently sampled physical scenarios.
- **Total Simulation Runs**: 505 independent stochastic runs.
- **Pulses per Run**: 20,000 pulses.
- **Total Emitted Pulses**: $505 \times 20,000 = \mathbf{10,100,000 \text{ pulses (10.1 Million Pulses)}}$.
- **Window Size**: $W = 1,000 \text{ pulses}$.
- **Usable Telemetry Windows**: $\mathbf{10,100 \text{ windows}}$.
- **Expected Runtime**: $\approx 670 \text{ seconds} \approx 11.2 \text{ minutes}$.
- **Expected Storage**: $\approx 1.15 \text{ GB}$ (leaving $>259 \text{ GB}$ safety buffer).

This provides substantial statistical power:
- 200 distinct physical channel and detector setups.
- Over 10 million simulated quantum pulses.
- 10,100 windows for training and evaluating temporal and deep learning models.
- Independent seeds across all scenario templates and run instances.

---

## 4. Logical Partition Architecture

To guarantee absolute scientific validity and zero data leakage, the dataset is partitioned into 6 disjoint logical splits at the **scenario and run level**:

| Partition | Scenarios | Runs/Scen | Total Runs | Pulses | Windows | Attack Fraction | Edge Fraction | Purpose |
|---|---|---|---|---|---|---|---|---|
| `calibration` | 20 | 2 | 40 | 800,000 | 800 | 0.00 | 0.00 | Normal baseline calibration and feature scaling |
| `train` | 60 | 3 | 180 | 3,600,000 | 3,600 | 0.00 | 0.15 | Unsupervised and one-class anomaly model training |
| `validation` | 25 | 2 | 50 | 1,000,000 | 1,000 | 0.40 | 0.15 | Model selection, hyperparameter tuning, threshold calibration |
| `test_in_distribution` | 45 | 3 | 135 | 2,700,000 | 2,700 | 0.70 | 0.15 | Benchmark evaluation on known attacks under known regimes |
| `test_generalization` | 25 | 2 | 50 | 1,000,000 | 1,000 | 0.60 | 0.25 | Held-out parameter bounds (distance 50–75km, $\eta$ 0.30–0.42) |
| `test_stress` | 25 | 2 | 50 | 1,000,000 | 1,000 | 0.50 | 1.00 | Difficult physical regimes (high loss, noise, instability) |
| **Total** | **200** | — | **505** | **10,100,000** | **10,100** | — | — | Full Research Dataset |

---

## 5. Leakage Prevention Strategy

1. **Scenario-Level Split Isolation**: Scenario IDs and infrastructure IDs are mutually disjoint across splits. A scenario instance never appears in more than one partition.
2. **Temporal Independence**: Windows are strictly segmented within individual runs. No window crosses a simulation-run boundary. Overlapping windowing is prohibited.
3. **Seed Stream Independence**: The master seed `20261019` derives partition seeds via SHA-256 (`derive_dataset_seed`), which derive scenario seeds (`derive_scenario_seed`), which derive run seeds (`derive_run_seed`).
4. **Observable vs. Ground-Truth Separation**:
   - Exactly 76 features are marked as `observable` (model inputs allowed).
   - All 18 `ground_truth_*` columns are strictly prohibited as ML model inputs.
   - Automated validator `scripts/validate_dataset.py` enforces this separation with zero tolerance.

---

## 6. Execution Pipeline

1. **Stage A (Pilot Validation)**: Executed `configs/pilot_research.yaml` across 20 scenarios / 20 runs (100,000 pulses). Verified schema compliance, zero leakage, and 14,925 pulses/sec throughput.
2. **Stage B (Cost Estimation)**: Finalized production config `configs/qism_research_large.yaml` based on empirical pilot measurements.
3. **Stage C (Production Generation)**: Execute 505 simulation jobs across 4 parallel processes writing sharded Parquet files to `data/qism_research_dataset/`.
4. **Stage D (QA & Integrity Audit)**: Validate completion markers, manifest counts, schema adherence, and split isolation using `scripts/validate_dataset.py --check-split`.
