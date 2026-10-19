# QISM Research Prototype: Comprehensive Experiment Reproduction Guide (V0.6–V1.0)

**Project:** Quantum Infrastructure Security Monitor (QISM)  
**Milestones Covered:** V0.6 (Point ML Baselines), V0.7 (Temporal & Sequential Models), V0.8 (Concept Drift & Adaptation), V0.9 (Integrated Incident Intelligence), V1.0 (Hardened Multi-Seed Release)  
**Target Environment:** Windows 10/11 or Linux x86_64, Python 3.10  

---

## 1. System Requirements & Environment Setup

### 1.1 Python Interpreter & Virtual Environment
QISM requires **Python 3.10** (specifically verified on Python 3.10.11 64-bit). The active virtual environment is located at `.venv/` within the repository root.

```bash
# 1. Verify system Python 3.10
python --version
# Expected: Python 3.10.x

# 2. Activate the pre-configured project virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS bash:
source .venv/bin/activate

# 3. Verify editable installation of qism package
python -c "import qism; print(qism.__version__)"
# Expected: 0.6.0 (or current package milestone)
```

### 1.2 Core Dependencies & Versions
The reproduction environment relies on standard, pin-compatible scientific Python libraries:
- `numpy >= 1.22.0`
- `scipy >= 1.9.0`
- `pandas >= 1.5.0`
- `pyarrow >= 10.0.0` (for sharded Parquet dataset streaming)
- `scikit-learn >= 1.2.0` (Isolation Forest, One-Class SVM, LOF, MinCovDet)
- `torch >= 2.0.0` (MLP and Recurrent GRU/LSTM Autoencoders)
- `matplotlib >= 3.6.0` and `seaborn >= 0.12.0` (Visualization suite)
- `pytest >= 7.2.0` (Unit and integration test suites)
- `pydantic >= 2.0.0` (Configuration validation schemas)

To reinstall or refresh all dependencies in a clean virtual environment:
```bash
pip install --upgrade pip
pip install -e .
pip install pytest matplotlib seaborn scikit-learn torch pandas pyarrow pydantic
```

---

## 2. Running Regression & Hardening Tests

To verify that the entire codebase across all milestones is functional and passes all invariants before running experiments:

```bash
# Run the complete test suite (245 tests across V0.1–V1.0)
python -m pytest -v

# Run only V1.0 hardening and reproducibility test suites
python -m pytest tests/test_v10_hardening.py tests/test_v10_reproducibility.py -v

# Run the leakage and security boundary tests
python -m pytest tests/test_v09_ingestion.py tests/test_v08_security.py -v
```

**Expected Test Output:**
- `245 passed in ~190s (100% pass rate, 0 failures, 0 errors)`.

---

## 3. Exact Commands to Reproduce Key Benchmarks

### 3.1 Reproducing the Canonical End-to-End Demo (`v10-demo`)
The canonical demonstration runs the integrated monitoring pipeline over a realistic sequential scenario featuring benign optical loss drift followed by an active Intercept-Resend attack.

```bash
# Execute canonical V1.0 demo (stores into artifacts/audit_reproduction/demo)
.venv\Scripts\python.exe -m qism.cli v10-demo --scenario drift_then_attack --outdir artifacts/audit_reproduction/demo --seed 42 --verbose
```

**Expected Console Output & Metrics:**
- Total Steps Evaluated: `120`
- Raw Alarms Triggered: `93`
- Correlated Incidents: `1`
- Emitted Alerts: `6`
- Alert Compression Ratio: `93.0x` (93.5% alert reduction)
- Incident Precision: `1.0000`, Incident Recall: `1.0000`, Incident F1: `1.0000`
- Mean Step Latency: `~5.5 to 6.0 ms` (Throughput: `~170 to 180 steps/s`)

**Generated Artifacts:**
- `artifacts/audit_reproduction/demo/plots/demo_timeline.png` (Aligned 4-panel timeline)
- `artifacts/audit_reproduction/demo/demo_dashboard.html` (Interactive web dashboard)
- `artifacts/audit_reproduction/demo/demo_audit_log.jsonl` (Append-only audit trail)

---

### 3.2 Reproducing the V1.0 Master Multi-Seed Benchmark Matrix (`v10-benchmark`)
This reproduces the full 12-scenario evaluation matrix across three random seeds (42, 101, and 2024), yielding 36 total runs, along with all five formal ablation comparisons (A through E).

```bash
# Run master benchmark suite across seeds 42, 101, 2024
.venv\Scripts\python.exe -m qism.cli v10-benchmark --outdir outputs/v1_0 --seeds 42,101,2024
```

**Expected Runtime:** `~35 to 45 seconds`.

**Generated Output Files in `outputs/v1_0/`:**
- `manifests/v10_benchmark_manifest.json` (Full execution metadata and run catalogue)
- `metrics/v10_benchmark_summary.csv` (Aggregated Mean ± Std table)
- `metrics/v10_benchmark_raw_seeds.csv` (Raw metrics for all 36 runs)
- `metrics/v10_ablations.json` (Structured results for Comparisons A, B, C, D, E)
- `audit_logs/v10_audit_log.jsonl` (Append-only JSONL audit events)
- `plots/` (13 publication-quality PNG charts, 300 DPI)
- `dashboard.html` (Complete interactive dashboard)

---

### 3.3 Reproducing the V0.9 Integrated Monitoring Benchmark (`v09-benchmark`)
To rerun the V0.9 integrated suite across its 12 scenarios on the baseline seed 42:

```bash
# Execute V0.9 benchmark runner
.venv\Scripts\python.exe -m qism.cli v09-benchmark --outdir outputs/v09/experiments
```

**Expected Output Files in `outputs/v09/experiments/`:**
- `summary.csv` (12 rows detailing compression ratios, latency, and incident metrics)
- `manifest.json` (Metadata, timing, and parameters)
- `plots/exp01_stable_timeline.png` through `exp12_ood_generalization_timeline.png`
- `dashboard.html` (Interactive V0.9 dashboard)

---

### 3.4 Reproducing the V0.8 Concept Drift & Safe Adaptation Suite (`v08-benchmark`)
To rerun the 12 drift scenarios, metric comparisons (Wasserstein vs KS vs PSI), window size ablations, and the critical contamination stress test:

```bash
# Execute V0.8 drift and adaptation benchmark
.venv\Scripts\python.exe -m qism.cli v08-benchmark --outdir outputs/v08/experiments --seed 42 --steps 120
```

**Expected Output Files in `outputs/v08/experiments/`:**
- `summary.csv` (18 rows covering safe, fixed, and naive adaptation policies)
- `manifest.json` (Complete run parameters, drift delays, and model versions)
- `registry/` (Transactional model versions: `model_v001`, `model_v002`)
- `plots/exp08_contamination_stress_timeline.png` (Visualizing naive poisoning vs safe gate)
- `plots/distribution_comparison.png` and `drift_heatmap.png`

---

### 3.5 Reproducing the V0.7 Temporal & Sequential Anomaly Benchmark (`v07-benchmark`)
To rerun the temporal sequence autoencoder, EWMA, rolling Z-score, and alarm hysteresis state machine experiments:

```bash
# Execute V0.7 temporal benchmark runner
.venv\Scripts\python.exe -m qism.cli v07-benchmark --dataset data/v0.5_small/v05_small_dataset --outdir outputs/v07/experiments --seed 42 --window-length 5 --epochs 10
```

**Expected Output Files in `outputs/v07/experiments/`:**
- `summary.csv` (Comparison table across Experiments 1 through 9)
- `manifest.json` (Profiling stats, top contributing features $E_j$)
- `plots/roc_pr_comparison.png` (Point baseline vs GRU-AE vs EWMA)
- `plots/feature_decomposition.png` (Reconstruction residual decomposition)
- `plots/window_length_study.png` ($L \in [2, 3, 5]$ performance curves)

---

### 3.6 Reproducing the V0.6 Baseline ML/DL Anomaly Benchmark
To rerun the foundational point anomaly detectors on `v05_small_dataset`:

```bash
# Execute V0.6 benchmark script
.venv\Scripts\python.exe scripts/run_v06_benchmark.py --dataset data/v0.5_small/v05_small_dataset --output reports
```

**Expected Output Files:**
- `reports/v06_benchmark_summary.json` (Consolidated metrics for all 7 models + ablations)
- `reports/v06_benchmark_summary.md` (Markdown summary table)
- Individual experiment directories in `experiments/exp_*/`:
  - `model.bin` (Saved model checkpoint)
  - `preprocessor.json` (Fitted Robust/Standard scaler parameters)
  - `results.json` (Raw window, curve, instance, and OOD metrics)
  - `summary.csv`
  - `plots/roc_curve.png`, `pr_curve.png`, `score_distributions.png`, `anomaly_timeline.png`

---

### 3.7 Regenerating Presentation & Physics Visualizations
To regenerate the 15 presentation-quality diagrams (PDF, PNG, SVG) and the exploratory diagnostics:

```bash
# Generate presentation figures
.venv\Scripts\python.exe scripts/generate_visualizations.py --dataset data/v0.5_small/v05_small_dataset --outdir outputs/visualizations --formats png,svg,pdf --dpi 300
```

**Expected Output Files in `outputs/visualizations/presentation/`:**
- `01_simulation_pipeline.png` through `15_stochastic_consistency_and_reproducibility.png` (alongside `.svg` and `.pdf` vectors).
- `outputs/visualizations/metadata/figure_metadata.json`.

---

## 4. Deterministic Random Seeding & PRNG Invariants

To guarantee exact bit-level numerical reproducibility:
1. **Master PRNG Seeding:** All top-level CLI commands accept explicit `--seed <INT>` arguments.
2. **Hierarchical Determinism:** Internal PRNG streams use `qism.random.derive_dataset_seed(master_seed, component_name)` to decouple subsystems (e.g. Monte Carlo photon sampling, Latin Hypercube parameter generation, PyTorch weight initialization, scikit-learn estimators).
3. **PyTorch Determinism:**
   ```python
   torch.manual_seed(seed)
   np.random.seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)
   ```
4. **Reproducibility Guarantee:** Repeating any benchmark with the same random seed produces **identical floating-point scores, threshold values, incident IDs, and audit log entries**.

---

## 5. Experiments That Cannot Currently Be Reproduced & Known Constraints

| Experiment / Capability | Status | Reason / Technical Constraint |
|---|---|---|
| **Quantum Machine Learning (QML) Execution** | Not Executable | **Zero QML code or circuits were implemented.** There are no PennyLane or Qiskit circuits, quantum variational algorithms, or quantum hardware backends in the repository. |
| **Physical Hardware Deployment** | Not Executable | QISM is a simulation-based research prototype. There are no physical optical device drivers, FPGA interfaces, or cryptographic key deletion actuators implemented. |
| **Large-Scale Multi-Node Distributed Training** | Not Supported | Model training is strictly single-node (multi-process for dataset generation, single-process PyTorch CPU/GPU for autoencoder training). |
| **Real Field Telemetry Replay** | Unavailable | All datasets in `data/` were generated by the QISM stochastic physics simulator; no third-party field-deployed QKD link telemetry files exist in the repository. |
