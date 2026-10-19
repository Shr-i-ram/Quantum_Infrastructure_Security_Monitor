# Quantum Infrastructure Security Monitor (QISM)
## Final Academic Report — Artifact Discovery and Evidence Handover Manifest

**Document Name:** `QISM_FINAL_REPORT_ARTIFACT_MANIFEST.md`  
**Purpose:** Comprehensive, authoritative catalog of all report-quality visual assets, empirical data tables, model implementations, schema definitions, and reproducibility artifacts required for writing the final academic research report in LaTeX.  
**Auditor:** Senior Research Software Auditor (Quantum Machine Learning, Anomaly Detection, Scientific Computing)  
**Date:** October 20, 2026  
**Repository Target:** QISM 76-Observable Feature Research Benchmark (`data/qism_research_dataset`, `results/qism_model_benchmark`)  
**Audit Policy:** Read-only discovery and verification. Zero source modifications, zero model retraining, zero benchmark re-execution.

---

## 1. Executive Summary

This manifest identifies, verifies, and packages all scientifically trustworthy artifacts from the QISM repository for inclusion in the final LaTeX project.

### Key Handover Principles
1. **Strict Provenance Alignment:** Only artifacts derived from the authoritative 76-observable feature dataset (`data/qism_research_dataset`, Simulator 0.6.0, Schema 0.5.0, Master Seed 20261019, Config Hash `80189033257fdeedf018b3afac859e9ffe422b2fc5019b05c899674d5eeb8499`) and its corresponding benchmark run (`results/qism_model_benchmark`) are recommended for primary reporting.
2. **Exclusion of Stale Pilot Data:** All artifacts from the earlier 48-feature pilot run (`docs/PI_TQSVDD_BENCHMARK_RESULTS.md`, `artifacts/pi_tqsvdd/`) and pre-V0.6 Review 2 runs (`artifacts/qml_review2/`) are explicitly classified as `EXCLUDE`.
3. **Minimal Upload Package:** The full research dataset spans 3.78 GB due to 10.1M raw optical event records. However, the final academic report can be fully substantiated using a **compact, high-fidelity package (~15 MB)** consisting of 7 publication-quality vector/raster plots, 4 dataset diagnostic figures, 4 QKD physics schematics, 8 machine-readable metric tables, schema dictionaries, and core implementation files.
4. **Methodological Transparency:** All known implementation nuances—such as the supervised baseline validation data flow, Quantum Kernel SVM sample truncation, and the absence of an explicit physical-constraint loss—are documented with exact code references to ensure the academic report is scientifically defensible.

---

## 2. Authoritative Benchmark and Run Identification

| Attribute | Verified Value | Evidence File & Location |
| :--- | :--- | :--- |
| **Authoritative Benchmark Root** | `results/qism_model_benchmark/` | Directory structure, checkpoints, and tables |
| **Authoritative Dataset Root** | `data/qism_research_dataset/` | Root manifest, partition markers, parquet shards |
| **Simulator Version** | `0.6.0` | `data/qism_research_dataset/dataset_manifest.json` (L4) |
| **Dataset Schema Version** | `0.5.0` | `data/qism_research_dataset/dataset_manifest.json` (L3) |
| **Dataset Master Seed** | `20261019` | `data/qism_research_dataset/dataset_manifest.json` (L5) |
| **Dataset Configuration Hash** | `80189033257fdeedf018b3afac859e9ffe422b2fc5019b05c899674d5eeb8499` | `data/qism_research_dataset/dataset_manifest.json` (L6) |
| **Git Commit Reference** | `869f6cddc18bf1f3699b93198d938af78557b0ec` | `data/qism_research_dataset/dataset_manifest.json` (L10) |
| **Dataset Generation Timestamp** | `2026-10-18T19:39:39.852498+00:00` | `data/qism_research_dataset/dataset_manifest.json` (L7) |
| **Benchmark Execution Timestamp** | `2026-10-19 02:34:13 UTC` | File creation timestamps in `results/qism_model_benchmark/tables/` |
| **Benchmark Master Seed** | `42` | `scripts/run_complete_benchmark.py` (L132) |
| **Total Physical Simulation Runs** | `505` independent runs across `200` unique scenarios | `data/qism_research_dataset/dataset_manifest.json` (L20–L22) |
| **Total Raw Telemetry Windows** | `10,100` windows ($8,520$ normal, $1,580$ attack) | Parquet telemetry file row sums across 6 partitions |
| **Causal Temporal Windows ($L=3$)** | `9,700` windows ($8,120$ normal, $1,580$ attack) | Evaluated sliding sequences ($L=3$, dropping 2 warm-ups/scenario) |
| **Evaluated Model Configurations** | Exactly `30` completed models (1 failed run) | `results/qism_model_benchmark/tables/aggregate_metrics.csv` |
| **Evaluated Prediction Rows** | Exactly `90` rows ($30\text{ models} \times 3\text{ test partitions}$) | `results/qism_model_benchmark/tables/per_run_metrics.csv` |

---

## 3. Latest Verified Figures and Images

### 3.1 Primary Benchmark Evaluation Figures (P0 — Essential)
These figures were generated directly from the authoritative benchmark tables (`aggregate_metrics.csv`, `compute_cost_table.csv`) by `scripts/generate_publication_figures.py`. Both 300 DPI PNG and publication-grade vector PDF formats are available:

| Priority | Exact Repository Path | Format | Dimensions / Size | Description & Plotted Values | Report Section |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **P0** | `results/qism_model_benchmark/figures/comparisons/01_overall_auroc_comparison.pdf` (`.png`) | Vector PDF / PNG | $14\times 9$ in, 300 DPI (522 KB) | **Overall In-Distribution AUROC Comparison:** Horizontal bar chart of all 30 models ranked by In-Dist AUROC. Shows `ClassicalTemporalAutoencoder` (#1, 0.762), `QuantumKernelSVM_4Q` (#2, 0.757*), `ClassicalLogisticRegression` (#3, 0.741), `PITQSVDD_4Q_Amplitude` (#4, 0.715), `ClassicalAutoencoder` (#5, 0.713), down to `PITQSVDD_4Q_IQPZZ` (0.484). Color-coded by family. | Results: Master Leaderboard |
| **P0** | `results/qism_model_benchmark/figures/comparisons/02_recall_at_1pct_fpr_comparison.pdf` (`.png`) | Vector PDF / PNG | $14\times 8$ in, 300 DPI (485 KB) | **Operational Recall at 1% FPR:** Bar chart of detection recall when threshold is calibrated on normal validation data at target 1% FPR. Highlights top performers: `ClassicalLogisticRegression` (40.3%), `Mahalanobis_Full` (40.0%), `ClassicalAutoencoder` (39.8%), `IsolationForest_PCA6` (39.6%), and `PITQSVDD_4Q_Amplitude` (38.2%). | Results: Operational Metrics |
| **P0** | `results/qism_model_benchmark/figures/comparisons/03_performance_vs_compute_frontier.pdf` (`.png`) | Vector PDF / PNG | $10\times 7$ in, 300 DPI (381 KB) | **Performance vs Compute Cost Frontier:** Pareto scatter plot of In-Distribution AUROC vs Inference Latency per window (ms, log scale). Demonstrates the $2,000\times\text{--}8,000\times$ inference speed advantage of classical models ($1\text{--}7\,\mu\text{s}$) over simulated quantum circuits ($8\text{--}15\,\text{ms}$) and Quantum Kernel SVM ($188\,\text{ms}$). | Discussion: Computational Cost |
| **P0** | `results/qism_model_benchmark/figures/pitqsvdd/04_qubit_count_sweep.pdf` (`.png`) | Vector PDF / PNG | $14\times 6$ in, 300 DPI (285 KB) | **PI-TQSVDD Qubit Count Sweep (2, 4, 6, 8 Qubits):** Dual-panel plot. Left panel: AUROC across In-Distribution, Generalization (OOD), and Stress splits, demonstrating peak at 4 qubits (0.6751 ID, 0.6952 Gen). Right panel: Simulation training wall-clock time scaling from 11.4s (2Q) to 38.2s (8Q). | Results: Flagship Model Ablations |
| **P0** | `results/qism_model_benchmark/figures/pitqsvdd/05_encoding_comparison.pdf` (`.png`) | Vector PDF / PNG | $10\times 6$ in, 300 DPI (158 KB) | **PI-TQSVDD Data Encoding Sweep (4 Qubits):** Bar chart comparing quantum encodings on 4 qubits: Amplitude (0.715), Data Reuploading (0.680), Angle Ref (0.675), Feature Group (0.664), and IQP/ZZ Phase Map (0.484, collapsed below chance). | Results: Quantum Data Encodings |
| **P0** | `results/qism_model_benchmark/figures/pitqsvdd/06_temporal_ablation_comparison.pdf` (`.png`) | Vector PDF / PNG | $9\times 6$ in, 300 DPI (157 KB) | **Objective Function & Temporal Loss Ablation:** Bar chart comparing Full PI-TQSVDD ($\lambda_t=0.5$, 0.675) vs No Temporal Loss ($\lambda_t=0.0$, 0.645) vs Static Quantum SVDD baseline (0.640) vs Classical Deep SVDD (0.679). Demonstrates $+2.98\%$ AUROC benefit from causal temporal modeling. | Results: Temporal Modeling Impact |
| **P0** | `results/qism_model_benchmark/figures/generalization/07_generalization_robustness.pdf` (`.png`) | Vector PDF / PNG | $12\times 7$ in, 300 DPI (187 KB) | **Generalization Resilience Across Operating Regimes:** Grouped bar chart comparing In-Distribution vs Generalization (50–75 km, low $\eta$) vs Stress (loss drift, noise spikes) for key models (`PITQSVDD_4Q_Angle_Ref`, `PITQSVDD_8Q_Angle`, `IsolationForest_Full`, `ClassicalDeepSVDD`, `VQC_4Q`, `ClassicalAutoencoder`). | Results: Generalization & Stress |

### 3.2 Dataset Diagnostic Visualizations (P1 — Strongly Recommended)
Located in `data/qism_research_dataset/diagnostics/`, generated directly from the authoritative dataset:

| Priority | Exact Repository Path | Format | Size | Description & Plotted Values | Report Section |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **P1** | `data/qism_research_dataset/diagnostics/partition_window_counts.png` | PNG | 89 KB | Stacked bar chart showing total, normal, and attack window distributions across all 6 partitions (calibration: 800, train: 3600, val: 1000, test_id: 2700, test_gen: 1000, test_stress: 1000). | Methodology: Dataset Design |
| **P1** | `data/qism_research_dataset/diagnostics/attack_family_distribution.png` | PNG | 85 KB | Attack episode and window count breakdown across the 6 supported attack families (Intercept-Resend, PNS, Source Manipulation, Channel Loss, Detector Manipulation, Timing Delay). | Methodology: Attack Scenarios |
| **P1** | `data/qism_research_dataset/diagnostics/normal_vs_attack_distributions.png` | PNG | 99 KB | Kernel density estimates of key physical telemetry features (QBER, gains, dead time loss rate, dark counts) under nominal vs attack conditions. | Methodology: Physical Telemetry |
| **P2** | `data/qism_research_dataset/diagnostics/parameter_space_coverage.png` | PNG | 221 KB | Scatter coverage of physical transmission parameters (fiber distance vs channel loss vs detector efficiency) establishing simulation bounds. | Methodology: Simulation Setup |

### 3.3 QKD Physical Simulation Schematics (P1 — Strongly Recommended)
Located in `outputs/visualizations/presentation/`, available in vector `.pdf`, `.svg`, and raster `.png`:

| Priority | Exact Repository Path | Format | Size | Description & Conceptual Value | Report Section |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **P1** | `outputs/visualizations/presentation/01_simulation_pipeline.pdf` (`.png`, `.svg`) | Vector PDF / PNG / SVG | 55 KB | **QISM Simulation Architecture Schematic:** End-to-end causal simulation flow: Alice WCP laser driver $\to$ Quantum fiber channel $\to$ Eve physical intervention $\to$ Bob SPAD detection unit $\to$ Sifting $\to$ Windowed telemetry extraction. | Methodology: Simulator Architecture |
| **P1** | `outputs/visualizations/presentation/03_channel_transmission_and_loss.pdf` (`.png`, `.svg`) | Vector PDF / PNG / SVG | 38 KB | **Quantum Channel Loss & Detection Floor:** Beer-Lambert exponential attenuation law $T(L) = 10^{-(\alpha L + L_{\text{sys}})/10}$ and expected detection rate decaying into thermal dark count floor. | Methodology: Optical Channel Physics |
| **P1** | `outputs/visualizations/presentation/04_detector_physics_and_dead_time.pdf` (`.png`, `.svg`) | Vector PDF / PNG / SVG | 57 KB | **SPAD Detector Dynamics:** Nonparalyzable dead-time saturation curves, rate-dependent effective dead time model $\tau_{\text{eff}}$, and afterpulsing decay queue. | Methodology: Detector Modeling |
| **P1** | `outputs/visualizations/presentation/05_basis_selection_and_sifting_funnel.pdf` (`.png`, `.svg`) | Vector PDF / PNG / SVG | 43 KB | **BB84 Sifting Funnel & Basis Correlation Matrix:** Visualizes the reduction from raw 10k optical pulses to detected clicks, sifted key bits, and QBER error rates. | Methodology: QKD Protocol & Sifting |

---

## 4. Essential Dataset and Results Files (P0 — Essential)

These files provide the exact machine-readable numbers to populate LaTeX tables and verify provenance without uploading large binary datasets:

| Priority | Exact Repository Path | Format | Size / Rows | Description & Key Contents | Role in LaTeX Report |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **P0** | `results/qism_model_benchmark/tables/aggregate_metrics.csv` | CSV | 6.5 KB / 30 rows | Master benchmark results table. Columns: `model_id`, `family`, `auroc_in_dist`, `auprc_in_dist`, `recall_at_1pct_fpr_in_dist`, `incident_recall_in_dist`, `mean_detection_delay_windows`, `false_alarm_rate`, `auroc_generalization`, `auroc_stress`, `train_time_s`. | **Table 1: Master Comparative Leaderboard** |
| **P0** | `results/qism_model_benchmark/tables/per_run_metrics.csv` | CSV | 33.5 KB / 90 rows | Comprehensive per-partition evaluation records ($30\text{ models} \times 3\text{ test splits}$). Contains exact thresholds (`threshold_1pct`, `threshold_5pct`), F1-scores, FPRs, precisions, and wall-clock times. | **Table 2 & 3: Partition Robustness Tables** |
| **P0** | `results/qism_model_benchmark/tables/per_attack_metrics.csv` | CSV | 13.3 KB / 176 rows | Per-attack recall breakdown across all 6 attack families (`intercept_resend`, `pns`, `source_manipulation`, `channel_loss`, `detector_manipulation`, `timing`) at calibrated 1% FPR threshold. | **Table 4: Attack-Specific Detection Recall** |
| **P0** | `results/qism_model_benchmark/tables/pitqsvdd_ablation_table.csv` | CSV | 15.3 KB / 39 rows | Dedicated PI-TQSVDD ablation records covering qubit scaling (2Q, 4Q, 6Q, 8Q), data encodings, objective formulations, and multi-seed trials (42, 101, 2024, 777). | **Table 5: Flagship Model Ablation Table** |
| **P0** | `results/qism_model_benchmark/tables/compute_cost_table.csv` | CSV | 2.8 KB / 30 rows | Hardware accounting and compute cost table. Contains trainable parameter counts, qubit counts, circuit depths, training wall-clock time, and inference latency per window. | **Table 6: Computational Cost & Complexity** |
| **P0** | `data/qism_research_dataset/schema/feature_dictionary.csv` | CSV | 20.9 KB / 132 rows | Complete authoritative data dictionary for all 76 observable physical telemetry features, units, valid ranges, and ground-truth exclusion flags. | **Table 7: 76-Feature Allowlist Specification** |
| **P0** | `data/qism_research_dataset/dataset_manifest.json` | JSON | 922 bytes | Root dataset manifest verifying dataset provenance, simulator version 0.6.0, master seed 20261019, and configuration hash. | **Reproducibility & Provenance Section** |
| **P0** | `data/qism_research_dataset/diagnostics/quality_audit_summary.json` | JSON | 2.7 KB | Verification summary confirming zero partition leakage, scenario-level isolation, and 10,100 raw window accounting. | **Dataset Quality & Governance Section** |

---

## 5. Useful Supplementary Data Files (P1 / P2)

| Priority | Exact Repository Path | Format | Size | Description & Supplementary Value |
| :---: | :--- | :---: | :---: | :--- |
| **P1** | `results/qism_model_benchmark/tables/failed_runs_table.csv` | CSV | 297 bytes | Forensic execution log recording the failure of `PITQSVDD_4Q_ContextAware` (`AttributeError: init_centre`), essential for reporting transparent model accounting. |
| **P1** | `results/qism_model_benchmark/tables/experiment_registry.csv` | CSV | 11.0 KB | 90-row registry mapping experiment execution IDs to model configurations, random seeds, and checkpoint directories. |
| **P1** | `data/qism_research_dataset/schema/telemetry_schema.json` | JSON | 39.8 KB | Complete PyArrow/JSON schema definition for all physical telemetry columns. |
| **P1** | `data/qism_research_dataset/schema/label_schema.json` | JSON | 804 bytes | Schema definition for ground-truth attack attributes and provenance tags. |
| **P2** | `data/qism_research_dataset/diagnostics/distribution_summary.csv` | CSV | 10.8 KB | Summary statistics (mean, std, min, percentiles) for observable telemetry features under normal training conditions. |
| **P2** | `data/qism_research_dataset/diagnostics/parameter_coverage.csv` | CSV | 3.0 KB | Distribution of physical simulation parameters (distance, loss, dark counts, dead times) across generated scenarios. |
| **P1** | `docs/QISM_IMPLEMENTATION_AUDIT.md` | Markdown | ~35 KB | The comprehensive research software implementation audit establishing all verified facts, mathematical equations, and methodology nuances. |

---

## 6. Implementation Files Needed for Methodology Writing

These source files represent the ground-truth code governing the benchmark. Review them to write accurate, reproducible methodology sections:

### 6.1 Benchmark Engine & Data Pipeline
- [`scripts/run_complete_benchmark.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/scripts/run_complete_benchmark.py):
  - `load_data` (L172–L211): Partition loading, metadata separation, 100% normal train assertion.
  - `fit_preprocessors` (L212–L294): Infilling all-NaN decoy residuals with 0.0, fitting `SimpleImputer`, `StandardScaler`, `MinMaxScaler(0, pi)`, and 5 PCA models strictly on train.
  - `_build_temporal_windows` (L295–L373): Causal sequence construction ($L=3$, dropping 2 warm-ups/scenario), next-step target assignment, `has_target` masking.
  - `_eval_and_record` (L374–L495): Validation threshold calibration at 1% and 5% FPR, evaluation across 3 test splits, metric computation.
  - `run_classical_supervised_baselines` (L668–L717): Supervised training construction using 500 normal train windows and 174 validation attack windows.
  - `run_pitqsvdd_ablation_grid` (L825–L958): Execution loop for all 14 PI-TQSVDD ablation configurations.
- [`scripts/generate_publication_figures.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/scripts/generate_publication_figures.py):
  - Exact plotting code for Figures 01 through 07, styling, color palettes, and axis limits.
- [`src/qism/telemetry/features.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/src/qism/telemetry/features.py):
  - `get_observable_feature_names` (L571–L623): Canonical 76-feature allowlist definition.
  - `_detector_dynamics_features` (L276–L454): SPAD dead-time loss rate, busy/ready fractions, interarrival moments.
  - `_infrastructure_features` (L456–L528): Sensor environmental stats, channel transmittance variance, optical loss estimates.

### 6.2 Flagship PI-TQSVDD Architecture & Circuits
- [`qml/pitqsvdd/models/temporal_pitqsvdd.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/pitqsvdd/models/temporal_pitqsvdd.py):
  - `TemporalPITQSVDD`: Hypersphere centre $c$ initialization buffer (empirical batch mean, perturbation if $<10^{-4}$ to $0.1$, frozen), static SVDD loss, temporal latent loss, continuous anomaly score:
    $$s_t = 0.5 \cdot \|z_t - c\|_2^2 + 0.5 \cdot \|z_t - \hat{z}_t\|_2^2$$
- [`qml/pitqsvdd/circuits/variational.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/pitqsvdd/circuits/variational.py):
  - `VariationalQuantumCircuit`: PennyLane `lightning.qubit` QNode with `diff_method="adjoint"`, RX-RY-RZ rotational layers, periodic CNOT ring entanglement, Pauli-$Z$ observable expectation measurements.
- [`qml/pitqsvdd/models/temporal_predictor.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/pitqsvdd/models/temporal_predictor.py):
  - `LatentTemporalPredictor`: 1-layer GRU (hidden dim 16) with Linear + Tanh head predicting next latent quantum state $z_{t+1}$ from history. Hardcoded internal `seed=42`.
- [`qml/pitqsvdd/encodings/amplitude.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/pitqsvdd/encodings/amplitude.py):
  - `AmplitudeEncoding`: Normalizes 16-feature PCA vector to unit sphere ($\|\psi\|_2 = 1$). Note: discards absolute vector norm.
- [`qml/pitqsvdd/encodings/data_reuploading.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/pitqsvdd/encodings/data_reuploading.py):
  - `DataReuploadingEncoding`: Slices input features across layers. Note: for 4Q and 2 layers, only first 8 of 16 features are injected.
- [`qml/pitqsvdd/encodings/iqp_zz.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/pitqsvdd/encodings/iqp_zz.py):
  - `IQPZZEncoding`: Hadamard gates, RZ rotations, and pairwise CNOT-RZ-CNOT phase interactions.
- [`qml/pitqsvdd/encodings/feature_group.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/pitqsvdd/encodings/feature_group.py):
  - `FeatureGroupEncoding`: Assigns PCA features to dedicated wires with ring CNOTs.

### 6.3 Baseline Implementations
- [`qml/src/vqc_model.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/src/vqc_model.py): Variational Quantum Classifier (4 qubits, 2 layers).
- [`qml/src/quantum_kernel_svm.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/src/quantum_kernel_svm.py): Quantum Kernel SVM with fidelity Gram matrix. Evaluated with `max_samples=150`.
- [`qml/src/quantum_autoencoder.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/src/quantum_autoencoder.py): Quantum Autoencoder (4 qubits $\to$ 2 latent, 2 trash wires).
- [`qml/pitqsvdd/models/static_qsvdd.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/pitqsvdd/models/static_qsvdd.py): Static Quantum SVDD baseline ($\beta=0.0$).
- [`qml/src/baselines.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/src/baselines.py): Small Classical MLP ($4 \to 8 \to 1$).

### 6.4 Operational Metrics Engine
- [`qml/pitqsvdd/evaluation/event_metrics.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/qml/pitqsvdd/evaluation/event_metrics.py):
  - `compute_event_level_metrics`: Episode grouping by `scenario_id`, attack onset tracking, detection delay calculation, false alarm counting over uncompromised windows.

---

## 7. Existing Architecture Diagrams

The repository contains several existing diagrams that can be referenced or adapted:

1. **`outputs/visualizations/presentation/01_simulation_pipeline.png` (and `.pdf`, `.svg`):**
   - *Status:* **Verified High Quality**.
   - *Description:* Complete vector dataflow diagram showing Alice $\to$ Channel $\to$ Bob $\to$ Sifting $\to$ Telemetry.
   - *Suitability:* Excellent for the QKD Simulator Architecture section of the report.
2. **`outputs/visualizations/presentation/05_basis_selection_and_sifting_funnel.png` (and `.pdf`, `.svg`):**
   - *Status:* **Verified High Quality**.
   - *Description:* Sifting funnel reduction from raw pulses to detected clicks, sifted key bits, and QBER.
   - *Suitability:* Excellent for the QKD Protocol and Sifting section.
3. **Mermaid 13-Stage Pipeline Diagram (`docs/QISM_V0_6_TO_V1_0_COMPREHENSIVE_TECHNICAL_REPORT.md`, L150–L188):**
   - *Status:* **Verified Accurate**.
   - *Description:* Flowchart depicting Physical Simulation $\to$ Ingestion $\to$ Inference $\to$ Incident Intelligence $\to$ Audit Logging.
   - *Suitability:* Can be rendered directly or exported to TikZ for an end-to-end overview.
4. **ASCII Architecture Flowchart (`docs/PI_TQSVDD_TECHNICAL_REPORT.md`, L17–L40):**
   - *Status:* **Verified Accurate**.
   - *Description:* Depicts sliding sequence input $\to$ quantum encoding $\to$ variational unitary $\to$ Pauli-Z measurement $\to$ static SVDD & causal GRU predictor $\to$ combined anomaly score.
   - *Suitability:* Serves as the precise structural blueprint for creating the final LaTeX PI-TQSVDD architecture diagram.

---

## 8. Recommended Diagrams to Create for the Final Report

While the existing figures cover experimental results, three dedicated diagrams should be created in TikZ or drawing software for the academic report:

### Diagram A: End-to-End QISM Research Pipeline (P0 — Essential)
- **Report Section:** System Architecture / Methodology Overview.
- **Purpose:** Illustrates the complete progression from physical optical simulation to decision thresholds and security alerts.
- **Required Blocks:**
  1. *Physical QKD Simulation:* Alice Laser Driver $\to$ Quantum Channel $\to$ Eve Intervention $\to$ Bob SPAD Detectors.
  2. *Telemetry Generation:* Event Aggregation $\to$ 76 Observable Physical Telemetry Features.
  3. *Leakage Governance:* Stripping 18 Ground-Truth Fields $\to$ Scenario-Disjoint Partitioning (6 Splits).
  4. *Preprocessing Pipeline:* Normal Train Median Imputation $\to$ StandardScaler $\to$ PCA $\to$ MinMaxScaler($0, \pi$).
  5. *Parallel Model Evaluation:* Classical Anomaly Detectors vs Supervised Baselines vs Quantum ML vs Flagship PI-TQSVDD.
  6. *Operational Calibration & Evaluation:* Normal Validation 1% FPR Threshold Calibration $\to$ 3 Test Splits (In-Dist, Gen, Stress) $\to$ Point & Event Metrics.
- **Mathematical Annotations:** $X_t \in \mathbb{R}^{76}$, $W_t \in \mathbb{R}^{L \times d}$, $\tau_{1\%} = \text{Percentile}(S_{\text{val\_norm}}, 99.0)$.

### Diagram B: Detailed PI-TQSVDD Model Architecture (P0 — Essential)
- **Report Section:** Proposed Method: Flagship PI-TQSVDD.
- **Purpose:** Rigorous mathematical and circuit-level depiction of the flagship quantum anomaly detector.
- **Required Blocks:**
  1. *Causal Window Tensor:* $X_t = [x_{t-2}, x_{t-1}, x_t] \in \mathbb{R}^{3 \times 4}$ (PCA-4 or Amplitude-16).
  2. *Quantum State Encoding:* Unitary $U_{\text{enc}}(x_\tau)$ mapping classical telemetry to $n$-qubit register ($|0\rangle^{\otimes n}$).
  3. *Parameterized Variational Unitary:* $M=2$ layers of single-qubit rotations ($R_X, R_Y, R_Z$) interleaved with periodic CNOT ring entanglement.
  4. *Observables Measurement:* Expectation values $\langle Z_j \rangle$ on all wires producing latent trajectory $Z_t \in [-1, 1]^{3 \times n}$.
  5. *Dual Objective Branches:*
     - *Branch 1 (Static SVDD):* Distance $\|z_t - c\|_2^2$ to fixed hypersphere centre $c \in \mathbb{R}^n$.
     - *Branch 2 (Causal GRU Predictor):* Recurrent GRU $g_\psi(Z_t)$ predicting next latent quantum state $\hat{z}_{t+1}$.
  6. *Joint Loss Function:* $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SVDD}} + \lambda_t \mathcal{L}_{\text{temporal}} + \lambda_{\text{reg}} \mathcal{R}$.
  7. *Inference Scoring:* Combined continuous score $s_t = 0.5 s_{\text{normal}} + 0.5 s_{\text{temporal}}$ compared against frozen threshold $\tau_{1\%}$.

### Diagram C: Experimental Partitioning and Scenario Isolation Matrix (P1 — Recommended)
- **Report Section:** Experimental Setup / Dataset Design.
- **Purpose:** Visualizes the 6 partitions, scenario counts (200 total), run counts (505 total), window counts, and parameter distribution shifts.
- **Required Blocks:**
  - Table/Matrix showing `calibration` (nominal), `train` (100% normal), `validation` (threshold tuning), `test_in_distribution` (all 6 attack types), `test_generalization` (50–75 km distance, $\eta \in [0.30, 0.42]$), and `test_stress` (extreme loss drift, thermal instability).
  - Explicit visual callout showing how length-3 causal sequences drop 2 warm-up windows per scenario ($2,700 \to 2,610$ in-dist, $1,000 \to 950$ OOD) while preserving 100% of attack ground-truth windows.

---

## 9. Artifacts and Figures That Must Be Excluded (EXCLUDE)

To protect the scientific credibility of the final report, the following files must **NOT** be included or cited:

| Exact Repository Path | Artifact Type | Reason for Exclusion |
| :--- | :--- | :--- |
| `docs/PI_TQSVDD_BENCHMARK_RESULTS.md` | Document | **Obsolete pilot benchmark** based on an early 48-feature dataset. Its metrics and conclusions conflict with the authoritative 76-feature benchmark. |
| `artifacts/pi_tqsvdd/` (all files) | Checkpoints & Figures | Pilot exploratory artifacts (e.g. `pilot_4q_amplitude`, `smoke_test_cli`) evaluated on preliminary datasets. |
| `artifacts/qml_review2/` & `artifacts/qml_review2_physical/` | Figures & Tables | Review 2 presentation figures from September 2026 evaluated on older pre-V0.6 feature configurations. |
| `reports/figures/01_poisson_photon_number.png` .. `20_multi_attack_progression.png` | PNG Plots | Early V0.1/V0.2 physics exploratory figures generated on September 5, 2026. Superseded by presentation figures. |
| `outputs/visualizations/presentation/06_normal_telemetry_distributions.png` | PNG Plot | Generated from `v05_small_dataset` on September 8, 2026. Superseded by `data/qism_research_dataset/diagnostics/`. |
| `outputs/visualizations/presentation/11_telemetry_cross_correlation.png` | PNG Plot | Generated from `v05_small_dataset` on September 8, 2026. Superseded by 76-feature correlation diagnostics. |
| `outputs/visualizations/presentation/12_dataset_partition_composition.png` | PNG Plot | Depicts obsolete V0.5 partition counts. Superseded by `partition_window_counts.png`. |
| `outputs/v08/`, `outputs/v1_0/` (all files) | Tables & Plots | Dynamic concept drift and streaming adaptation pipeline artifacts, outside the static 76-feature benchmark scope. |
| `Project_Sim.pdf` (5.3 MB) | PDF | Preliminary project proposal/report from an earlier milestone. |
| `qism-live-demo/` (all files) | Web Application | Frontend and dashboard code, explicitly outside the research benchmark scope. |

---

## 10. Known Discrepancies and Methodological Disclosures

The academic report must transparently disclose the following implementation details:

1. **Supervised Baseline Validation Compromise:**
   Supervised baselines (`ClassicalLogisticRegression`, `RandomForestClassifier`, `SmallClassicalMLP`, `ClassicalRBFSVM`, `VQC_4Q`, `QuantumKernelSVM_4Q`) were trained using the 174 attack windows from `validation`. Later, decision thresholds were calibrated using the normal windows of that same partition. While test sets remained strictly held-out, validation was not an independent split for supervised baselines.
2. **Quantum Kernel SVM Evaluation Truncation:**
   On the master leaderboard, `QuantumKernelSVM_4Q` ranks #2 with In-Dist AUROC $0.7570$. However, code inspection confirms it was evaluated on only 150 test windows (`max_samples=150`) due to quadratic Gram matrix scaling, encountering only 2 of the 6 attack types (`pns` and `channel_loss`). On stress, its AUROC collapsed to $0.2980$. It must be annotated with an asterisk denoting reduced-scale evaluation.
3. **Statistical Significance Claim ($p < 0.01$):**
   `QISM_BENCHMARK_RESULTS.md` Question 19 claims temporal modeling improvements are significant at $p < 0.01$ across scenario bootstrap resamples. The codebase contains no implementation of scenario bootstrap testing. This claim must either be removed or substantiated via an independent bootstrap test.
4. **Context-Aware PI-TQSVDD Runtime Failure:**
   `PITQSVDD_4Q_ContextAware` failed during benchmark execution due to an `AttributeError` (`init_centre` vs `init_centres`) and was recorded in `failed_runs_table.csv`. The model was not evaluated, leaving 30 completed models in `aggregate_metrics.csv`.
5. **Decoy-State Residual Missingness:**
   Three decoy residual columns (`signal_decoy_residual`, `decoy_decoy_residual`, `vacuum_residual`) are all-NaN in training data due to unconfigured yield analysis and are imputed to 0.0.
6. **Amplitude Encoding Magnitude Loss:**
   `AmplitudeEncoding` normalizes the 16-feature PCA vector to unit length ($\|\psi\|_2 = 1$), eliminating absolute vector magnitude information.
7. **4-Qubit Data Reuploading Feature Truncation:**
   For 4 qubits and 2 layers, `DataReuploadingEncoding` injects only the first 8 of 16 features across the two layers, leaving features 8–15 unused.
8. **PI-TQSVDD Training Duration:**
   To maintain tractable simulation times on classical CPUs, PI-TQSVDD models were trained on a 150-sequence slice of `train` for 8 gradient steps using Adam ($\text{lr}=0.02$).

---

## 11. Minimal Upload Package for LaTeX

To avoid transferring the 3.78 GB raw dataset, the following minimal file set (~15 MB) contains 100% of the evidence needed to write, format, and substantiate the academic report:

```
latex_handover_package/
├── figures/
│   ├── 01_overall_auroc_comparison.pdf
│   ├── 02_recall_at_1pct_fpr_comparison.pdf
│   ├── 03_performance_vs_compute_frontier.pdf
│   ├── 04_qubit_count_sweep.pdf
│   ├── 05_encoding_comparison.pdf
│   ├── 06_temporal_ablation_comparison.pdf
│   ├── 07_generalization_robustness.pdf
│   ├── 01_simulation_pipeline.pdf
│   ├── 03_channel_transmission_and_loss.pdf
│   ├── 04_detector_physics_and_dead_time.pdf
│   ├── 05_basis_selection_and_sifting_funnel.pdf
│   ├── partition_window_counts.png
│   ├── attack_family_distribution.png
│   └── normal_vs_attack_distributions.png
├── tables/
│   ├── aggregate_metrics.csv
│   ├── per_run_metrics.csv
│   ├── per_attack_metrics.csv
│   ├── pitqsvdd_ablation_table.csv
│   ├── compute_cost_table.csv
│   ├── failed_runs_table.csv
│   └── feature_dictionary.csv
├── metadata/
│   ├── dataset_manifest.json
│   ├── quality_audit_summary.json
│   ├── experiment_registry.csv
│   ├── telemetry_schema.json
│   └── label_schema.json
└── methodology_reference/
    ├── QISM_IMPLEMENTATION_AUDIT.md
    ├── run_complete_benchmark.py
    ├── generate_publication_figures.py
    ├── features.py
    ├── temporal_pitqsvdd.py
    ├── variational.py
    ├── temporal_predictor.py
    └── event_metrics.py
```

---

## 12. Suggested Directory Structure for the LaTeX Project

```
qism_final_report_latex/
├── main.tex
├── references.bib
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_qkd_simulation_and_telemetry.tex
│   ├── 04_methodology_pitqsvdd.tex
│   ├── 05_experimental_setup.tex
│   ├── 06_results_and_ablation.tex
│   ├── 07_discussion_and_limitations.tex
│   └── 08_conclusion.tex
├── figures/
│   ├── benchmark/        (PDF vector plots: Figs 01-07)
│   ├── physics/          (PDF vector plots: Simulation, SPAD, Loss)
│   ├── diagnostics/      (PNG dataset plots: Partitions, Attacks)
│   └── diagrams/         (TikZ source files for Diagrams A, B, C)
├── tables/
│   ├── leaderboard.tex   (Formatted from aggregate_metrics.csv)
│   ├── per_attack.tex    (Formatted from per_attack_metrics.csv)
│   ├── ablation.tex      (Formatted from pitqsvdd_ablation_table.csv)
│   ├── compute_cost.tex  (Formatted from compute_cost_table.csv)
│   └── features.tex      (Formatted from feature_dictionary.csv)
└── data_evidence/        (Raw CSV tables stored for archival/reproducibility)
```

---

## 13. Final Checklist of Files to Collect

Use this checklist when copying artifacts to the LaTeX workspace:

### Figures (14 files)
- [x] `results/qism_model_benchmark/figures/comparisons/01_overall_auroc_comparison.pdf`
- [x] `results/qism_model_benchmark/figures/comparisons/02_recall_at_1pct_fpr_comparison.pdf`
- [x] `results/qism_model_benchmark/figures/comparisons/03_performance_vs_compute_frontier.pdf`
- [x] `results/qism_model_benchmark/figures/pitqsvdd/04_qubit_count_sweep.pdf`
- [x] `results/qism_model_benchmark/figures/pitqsvdd/05_encoding_comparison.pdf`
- [x] `results/qism_model_benchmark/figures/pitqsvdd/06_temporal_ablation_comparison.pdf`
- [x] `results/qism_model_benchmark/figures/generalization/07_generalization_robustness.pdf`
- [x] `outputs/visualizations/presentation/01_simulation_pipeline.pdf`
- [x] `outputs/visualizations/presentation/03_channel_transmission_and_loss.pdf`
- [x] `outputs/visualizations/presentation/04_detector_physics_and_dead_time.pdf`
- [x] `outputs/visualizations/presentation/05_basis_selection_and_sifting_funnel.pdf`
- [x] `data/qism_research_dataset/diagnostics/partition_window_counts.png`
- [x] `data/qism_research_dataset/diagnostics/attack_family_distribution.png`
- [x] `data/qism_research_dataset/diagnostics/normal_vs_attack_distributions.png`

### Tables and Data (7 files)
- [x] `results/qism_model_benchmark/tables/aggregate_metrics.csv`
- [x] `results/qism_model_benchmark/tables/per_run_metrics.csv`
- [x] `results/qism_model_benchmark/tables/per_attack_metrics.csv`
- [x] `results/qism_model_benchmark/tables/pitqsvdd_ablation_table.csv`
- [x] `results/qism_model_benchmark/tables/compute_cost_table.csv`
- [x] `results/qism_model_benchmark/tables/failed_runs_table.csv`
- [x] `data/qism_research_dataset/schema/feature_dictionary.csv`

### Metadata and Provenance (5 files)
- [x] `data/qism_research_dataset/dataset_manifest.json`
- [x] `data/qism_research_dataset/diagnostics/quality_audit_summary.json`
- [x] `results/qism_model_benchmark/tables/experiment_registry.csv`
- [x] `data/qism_research_dataset/schema/telemetry_schema.json`
- [x] `data/qism_research_dataset/schema/label_schema.json`

### Methodology and Audit References (8 files)
- [x] `docs/QISM_IMPLEMENTATION_AUDIT.md`
- [x] `scripts/run_complete_benchmark.py`
- [x] `scripts/generate_publication_figures.py`
- [x] `src/qism/telemetry/features.py`
- [x] `qml/pitqsvdd/models/temporal_pitqsvdd.py`
- [x] `qml/pitqsvdd/circuits/variational.py`
- [x] `qml/pitqsvdd/models/temporal_predictor.py`
- [x] `qml/pitqsvdd/evaluation/event_metrics.py`

---
*Manifest certified by Senior Research Software Auditor.*
