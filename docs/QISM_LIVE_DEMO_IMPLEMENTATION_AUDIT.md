# Quantum Infrastructure Security Monitor (QISM)
## Live Demonstration Web Application — Implementation Audit

**Audit Date**: October 19, 2026  
**System Status**: Pre-Implementation Verified  
**Default Flagship Model**: **PI-TQSVDD — 6 Qubits** (`PITQSVDD_6Q_Angle.pt`)  
**Backend Framework**: FastAPI + Uvicorn + WebSockets + SQLite  
**Frontend Framework**: React 18 + TypeScript + Vite + High-Contrast Technical Console  

---

## 1. Executive Summary

This audit assesses all existing assets in the QISM repository prior to constructing the browser-based live demonstration web application. Every component—the decoy-state BB84 stochastic simulator, telemetry extraction schema, causal attack framework, preprocessors, and 30 trained model checkpoints—has been verified on the filesystem and smoke-tested for online inference.

---

## 2. Inventory of Reusable Repository Assets

### 2.1 QKD Simulator Entry Points
- **Primary Simulator Engine**: [`src/qism/simulation/simulator.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/src/qism/simulation/simulator.py)
  - Implements the chunked Monte Carlo event engine across Alice source, Poisson photon statistics, quantum channel attenuation, Bob passive basis selection, and SPD detector dynamics (avalanche, dead time, afterpulsing).
- **Configuration Engine**: [`src/qism/config.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/src/qism/config.py)
  - Loads YAML configurations defining optical parameters (attenuation $\alpha$, fiber length $L$, efficiency $\eta$, dark count rate $p_d$, timing jitter $\sigma_t$).
- **Telemetry Window Aggregator**: [`src/qism/telemetry/windows.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/src/qism/telemetry/windows.py) and [`src/qism/telemetry/features.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/src/qism/telemetry/features.py)
  - Computes 76 observable physical telemetry features strictly from event clicks without label leakage.

### 2.2 Causal Attack Engine
- **Attack Framework**: [`src/qism/attacks/base.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/src/qism/attacks/base.py)
  - Lifecycle: `INACTIVE -> ARMED -> ACTIVE -> TERMINATING -> COMPLETED`.
- **Supported Attack Implementations**: [`src/qism/attacks/intercept_resend.py`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/src/qism/attacks/intercept_resend.py)
  1. `InterceptResendAttack`: Parameterized by `fraction` and `eve_basis_prob_z`, causally raising QBER to $25\%$.
  2. `PNSAttack`: Photon-Number Splitting, selectively attenuating single photons and splitting multi-photons, distorting decoy yield ratios.
  3. `DetectorManipulationAttack`: Modifies detector efficiency mismatch and dark count rates.
  4. `ChannelLossAttack`: Introduces variable optical attenuation $\Delta L_{\text{max}}$, dropping throughput.
  5. `TimingAttack`: Shifts pulse arrival offset relative to Bob's detection gate, altering Fisher kurtosis and interarrival statistics.
  6. `SourceManipulationAttack`: Fluctuates mean photon intensity $\mu$.

### 2.3 Verified Model Checkpoints & Artifacts
All models are verified and located in [`results/qism_model_benchmark/checkpoints/`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/results/qism_model_benchmark/checkpoints/):

| Model Category | Checkpoint Identifier | File Path | Verified Status |
| :--- | :--- | :--- | :---: |
| **Default Flagship** | `PITQSVDD_6Q_Angle` | `checkpoints/pitqsvdd/PITQSVDD_6Q_Angle.pt` | **READY (Primary)** |
| Flagship Alternative | `PITQSVDD_4Q_Angle_Ref` | `checkpoints/pitqsvdd/PITQSVDD_4Q_Angle_Ref.pt` | READY |
| Flagship Alternative | `PITQSVDD_4Q_Amplitude` | `checkpoints/pitqsvdd/PITQSVDD_4Q_Amplitude.pt` | READY |
| Flagship Alternative | `PITQSVDD_4Q_DataReuploading` | `checkpoints/pitqsvdd/PITQSVDD_4Q_DataReuploading.pt` | READY |
| Flagship Alternative | `PITQSVDD_8Q_Angle` | `checkpoints/pitqsvdd/PITQSVDD_8Q_Angle.pt` | READY |
| Flagship Alternative | `PITQSVDD_8Q_DataReuploading` | `checkpoints/pitqsvdd/PITQSVDD_8Q_DataReuploading.pt` | READY |
| Classical Baseline | `ClassicalTemporalAutoencoder` | `checkpoints/classical/temporal_autoencoder.pt` | READY |
| Classical Baseline | `ClassicalAutoencoder` | `checkpoints/classical/autoencoder_dense.pt` | READY |
| Classical Baseline | `ClassicalDeepSVDD` | `checkpoints/classical/deep_svdd.pt` | READY |
| Classical Baseline | `IsolationForest_PCA6` | `checkpoints/classical/isolation_forest_pca6.joblib` | READY |
| Classical Baseline | `Mahalanobis_Full` | `checkpoints/classical/mahalanobis_full.npz` | READY |
| Quantum Baseline | `StaticQuantumSVDD_4Q` | `checkpoints/quantum/static_qsvdd_4q.pt` | READY |
| Quantum Baseline | `QuantumAutoencoder_4Q` | `checkpoints/quantum/quantum_autoencoder_4q.pt` | READY |
| Quantum Baseline | `VQC_4Q` | `checkpoints/quantum/vqc_4q.pt` | READY |
| Quantum Baseline | `QuantumKernelSVM_4Q` | `checkpoints/quantum/quantum_kernel_svm.joblib` | READY |
| Supervised Baseline | `ClassicalLogisticRegression` | `checkpoints/baselines/logistic_regression.joblib` | READY |

### 2.4 Preprocessor Pipeline
- **Saved Pipeline Path**: [`results/qism_model_benchmark/preprocessors/fitted_pipeline.joblib`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/results/qism_model_benchmark/preprocessors/fitted_pipeline.joblib)
- Contains:
  - `observable_cols`: 76 physical observable features.
  - `imputer`: SimpleImputer (median fitted on normal `train`).
  - `std_scaler`: StandardScaler.
  - `robust_scaler`: RobustScaler.
  - `angle_scaler`: MinMaxScaler $[0, \pi]$.
  - `pca_models`: Dict of fitted PCA models for $k \in \{2, 4, 6, 8, 16\}$.
  - `pca_scalers`: Dict of PCA Angle Scalers $[0, \pi]$.

### 2.5 Authoritative Research Replay Scenarios
Located in [`data/qism_research_dataset/test_in_distribution/`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/data/qism_research_dataset/test_in_distribution/):
- `shard_0000/attacks/run_0000` (PNS attack scenario, 20 windows)
- `shard_0000/attacks/run_0003` (Intercept-Resend attack scenario, 20 windows)
- `shard_0000/attacks/run_0008` (Timing attack scenario, 20 windows)
- `shard_0000/attacks/run_0009` (Detector manipulation attack scenario, 20 windows)
- `calibration/shard_0000/normal/run_0000` (Nominal baseline scenario, 100% normal)

### 2.6 Saved Benchmark Evaluation Metrics & Visualizations
- Master Tables: [`results/qism_model_benchmark/tables/`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/results/qism_model_benchmark/tables/)
  - `aggregate_metrics.csv`
  - `compute_cost_table.csv`
  - `per_attack_metrics.csv`
  - `pitqsvdd_ablation_table.csv`
- Publication Figures: [`results/qism_model_benchmark/figures/`](file:///c:/Users/Shriram/Desktop/python/QKD_Sim/qism-simulator/results/qism_model_benchmark/figures/)
  - `comparisons/01_overall_auroc_comparison.png` (& `.pdf`)
  - `comparisons/02_recall_at_1pct_fpr_comparison.png` (& `.pdf`)
  - `comparisons/03_performance_vs_compute_frontier.png` (& `.pdf`)
  - `pitqsvdd/04_qubit_count_sweep.png` (& `.pdf`)
  - `pitqsvdd/05_encoding_comparison.png` (& `.pdf`)
  - `pitqsvdd/06_temporal_ablation_comparison.png` (& `.pdf`)
  - `generalization/07_generalization_robustness.png` (& `.pdf`)

---

## 3. Web Application Architectural Blueprint

```text
Browser Client (React 18 + TypeScript + Modern Dark Console)
   ├── 1. Live Monitor (Header, Status Strip, Controls, Telemetry Charts, Score vs Threshold)
   ├── 2. Incidents (Filterable Incident Table, Evidence Snapshot Drawer, Lifecycle Actions)
   ├── 3. Telemetry Explorer (Multi-feature Inspector, Correlation Views, CSV/JSON Export)
   ├── 4. Model Lab (Model Selector, Live Comparison Canvas, Inference Latency Benchmark)
   ├── 5. Evaluation Results (Master Leaderboard, Benchmark Tables, Publication Figures)
   └── 6. System Health (Simulator Diagnostic, Quantum Engine Status, SQLite Store, Uptime)
           │
           │ REST API (JSON) + WebSocket Stream (/ws/telemetry)
           ▼
FastAPI Backend (qism-live-demo/backend/)
   ├── Simulation Controller (Step-by-step chunk generation & dynamic attack injection)
   ├── Replay Controller (Authoritative parquet replay with ground-truth alignment)
   ├── Inference Service (Loads PITQSVDD-6Q default + classical/quantum model pool)
   ├── Preprocessor Engine (Applies fitted median imputation + StandardScaler + PCA-6 + Angle)
   ├── Incident Manager (Consecutive-alarm policy, threshold comparison, severity mapping)
   └── SQLite Event Store (Sessions, Telemetry History, Incident Database, Audit Trail)
```

---

## 4. Verification Checkpoint

- [x] Checkpoint `results/qism_model_benchmark/checkpoints/pitqsvdd/PITQSVDD_6Q_Angle.pt` verified for forward pass.
- [x] Preprocessor pipeline `fitted_pipeline.joblib` built and verified.
- [x] Node.js and npm verified.
- [x] FastAPI, Uvicorn, and WebSockets installed in `.venv`.
- [x] Ground-truth vs. model prediction semantic separation established.
- [x] SQLite database schema designed for session and incident persistence.

**Audit Conclusion**: READY FOR IMPLEMENTATION.
