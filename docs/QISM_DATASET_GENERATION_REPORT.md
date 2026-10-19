# QISM Research Dataset Generation Report

## 1. Executive Summary

This report documents the completion of the production data-generation run for **`qism_research_dataset`**, produced by the **Quantum Infrastructure Security Monitor (QISM)** simulator (v0.6.0).
The dataset was generated deterministically across 200 independently sampled physical scenarios, partitioned across 6 disjoint logical splits, executed over 505 independent Monte Carlo runs, and sharded to Parquet format.

**Critical Milestone**:
- **Total Physical Quantum Pulses**: 10,100,000 pulses
- **Total Usable Telemetry Windows**: 10,100 windows ($W = 1,000 \text{ pulses/window}$)
- **Total Completed Simulation Runs**: 505 runs
- **Total Failed Runs**: 0
- **Total Independent Scenarios**: 200 scenarios
- **Total Storage Consumed**: 1,355.61 MB (~1.36 GB)
- **Total Generation Wall Time**: 599.06 seconds (~9.98 minutes)
- **Parallel Workers**: 4 processes
- **Effective Simulation Throughput**: 16,860 pulses/second

---

## 2. Partition Breakdown

The dataset enforces scenario-level and run-level split isolation across 6 required logical partitions:

| Partition | Scenarios | Runs | Usable Windows | Physical Pulses | Shards | Attack Fraction | Edge Fraction | Normal Windows | Attack Windows | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `calibration` | 20 | 40 | 800 | 800,000 | 3 | 0.00 | 0.00 | 800 | 0 | COMPLETED |
| `train` | 60 | 180 | 3,600 | 3,600,000 | 12 | 0.00 | 0.15 | 3,600 | 0 | COMPLETED |
| `validation` | 25 | 50 | 1,000 | 1,000,000 | 4 | 0.40 | 0.15 | 800 | 200 | COMPLETED |
| `test_in_distribution` | 45 | 135 | 2,700 | 2,700,000 | 9 | 0.70 | 0.15 | 1,940 | 760 | COMPLETED |
| `test_generalization` | 25 | 50 | 1,000 | 1,000,000 | 4 | 0.60 | 0.25 | 700 | 300 | COMPLETED |
| `test_stress` | 25 | 50 | 1,000 | 1,000,000 | 4 | 0.50 | 1.00 | 680 | 320 | COMPLETED |
| **Total** | **200** | **505** | **10,100** | **10,100,000** | **36** | — | — | **8,520** | **1,580** | **COMPLETED** |

---

## 3. Physical Operating Regimes

The Latin Hypercube sampling strategy, coupled with physical parameter dependency enforcement (`qism.population.dependencies.resolve_dependencies`), generated 5 distinct operational regimes across the 200 scenarios:

| Operating Regime | Scenario Count | Description |
|---|---|---|
| `nominal` | 130 | Standard laboratory baseline conditions (distance 10–45 km, standard fiber attenuation 0.20 dB/km, baseline dark counts ~1000 Hz). |
| `high_loss` | 31 | Extended channel distance (45–65 km) and elevated connector/coupling insertion losses (2.5–4.5 dB). |
| `low_signal` | 17 | Suppressed laser emission ($\mu_{\text{signal}} \in [0.15, 0.28]$) and lower detector efficiency ($\eta_d \times 0.6$). |
| `high_instability` | 15 | Elevated thermal drift rates and polarization scintillation fluctuations. |
| `high_noise` | 7 | High background stray light (400–900 lux) and elevated detector dark count rates (5,000–20,000 Hz). |

---

## 4. Attack Coverage & Window Distribution

All 6 physical attack mechanisms implemented in the QISM simulation engine were successfully generated across independent scenario instances, with continuous variations in attack severity $\xi \in [0.15, 0.95]$, randomized start pulses (15–40% of run), and smoothstep onset/recovery profiles:

| Attack Family | Scenarios | Attack Windows | Affected Physical Telemetry |
|---|---|---|---|
| `intercept_resend` | 24 | 555 | QBER elevation (~25% in matched basis), count rate parity, timing preservation. |
| `pns` (Photon Number Splitting) | 13 | 321 | Single-photon suppression, multi-photon enhancement, decoy residual shifts ($R_\mu, R_\nu$). |
| `channel_loss` | 11 | 245 | Insertion loss spike, count rate attenuation, estimated loss jump. |
| `source_manipulation` | 9 | 197 | Laser intensity perturbation ($\Delta\mu$), decoy gain divergence. |
| `detector_manipulation` | 8 | 184 | Efficiency mismatch ($\Delta\eta$) or dark count surge inducing strong `detector_imbalance`. |
| `timing` | 3 | 78 | Pulse delay shift outside detector gate center, `mean_arrival_offset` and `timing_skewness` anomalies. |
| **Total Attack** | **68** | **1,580** | Multi-channel anomalous physical telemetry. |

The remaining 132 scenarios (and remaining pulses in attack runs outside attack intervals) constitute **8,520 clean, normal telemetry windows**.

---

## 5. Artifact Inventory & Locations

All generated artifacts are persisted to disk and available without rerunning the simulator:

1. **Dataset Root**: `data/qism_research_dataset/`
2. **Top-Level Manifests**:
   - `data/qism_research_dataset/dataset_manifest.json`
   - `data/qism_research_dataset/manifest.json`
3. **Partition Manifests & Complete Markers**:
   - `data/qism_research_dataset/calibration/manifest.json`, `dataset_complete.marker`
   - `data/qism_research_dataset/train/manifest.json`, `dataset_complete.marker`
   - `data/qism_research_dataset/validation/manifest.json`, `dataset_complete.marker`
   - `data/qism_research_dataset/test_in_distribution/manifest.json`, `dataset_complete.marker`
   - `data/qism_research_dataset/test_generalization/manifest.json`, `dataset_complete.marker`
   - `data/qism_research_dataset/test_stress/manifest.json`, `dataset_complete.marker`
4. **Machine-Readable Schemas**:
   - `data/qism_research_dataset/schema/feature_dictionary.csv` (130 feature dictionary)
   - `data/qism_research_dataset/schema/telemetry_schema.json`
   - `data/qism_research_dataset/schema/label_schema.json`
5. **Scenario Registry**:
   - `data/qism_research_dataset/scenario_registry/scenario_registry.parquet` (and `.csv`)
   - `data/qism_research_dataset/scenario_registry/scenario_parameters.parquet` (and `.csv`)
6. **Diagnostics & QA Outputs**:
   - `data/qism_research_dataset/diagnostics/quality_audit_summary.json`
   - `data/qism_research_dataset/diagnostics/distribution_summary.csv`
   - `data/qism_research_dataset/diagnostics/parameter_coverage.csv`
   - `data/qism_research_dataset/diagnostics/parameter_coverage_hists.png`
   - `data/qism_research_dataset/diagnostics/parameter_correlation_matrix.png`
   - `data/qism_research_dataset/diagnostics/regime_distribution.png`
