# Quantum Infrastructure Security Monitor (QISM)
## Comprehensive Reproducibility and Environment Specification

This document provides complete provenance, software environment definitions, random seed hierarchies, configuration hashes, and reproduction workflows for the entire QISM research benchmark.

---

## 1. Software and Hardware Environment

All experiments were executed in a controlled, pinned environment:
- **Operating System**: Microsoft Windows 11 Enterprise (x86_64, Windows-10-10.0.26300-SP0).
- **Python Version**: Python 3.10.11 (CPython, 64-bit).
- **Core Runtime Dependencies**:
  - `numpy`: 2.2.6
  - `scipy`: 1.15.3
  - `pandas`: 2.2.3
  - `pyarrow`: 19.0.1
  - `scikit-learn`: 1.6.1
  - `torch`: 2.6.0+cpu
  - `pennylane`: 0.42.3
  - `pennylane-lightning`: 0.42.0 (`lightning.qubit` C++ state-vector plugin)
  - `matplotlib`: 3.10.0
  - `joblib`: 1.4.2
  - `pydantic`: 2.13.5

---

## 2. Dataset Provenance and Integrity

- **Authoritative Dataset Root**: `data/qism_research_dataset/`
- **Dataset ID**: `qism_research_dataset`
- **Schema Version**: `0.5.0`
- **Simulator Version**: `0.6.0`
- **Dataset Master Seed**: `20261019`
- **Dataset Configuration Hash**: `80189033257fdeedf018b3afac859e9ffe422b2fc5019b05c899674d5eeb8499`
- **Total Physical Simulation Runs**: `505` independent runs across `200` unique scenarios.
- **Total Observable Features**: `76` continuous numerical telemetry features.
- **Ground Truth Target Columns**: `18` ground-truth fields strictly isolated from model inputs.

---

## 3. Random Seed Hierarchy and Determinism

To guarantee reproducible model initializations and partition assignments:
1. **Benchmark Master Seed**: `42` (controls scikit-learn estimators, train/val development splits, and reference quantum weight initialization).
2. **Multi-Seed Stability Sweep**:
   - `Seed 42`: Reference initialization.
   - `Seed 101`: Multi-seed trial 2.
   - `Seed 2024`: Multi-seed trial 3.
   - `Seed 777`: Multi-seed trial 4.
3. **PyTorch Determinism**: `torch.manual_seed(seed)` is invoked prior to initializing each neural network, classical projection, and variational circuit ansatz.
4. **NumPy Determinism**: `np.random.RandomState(seed)` governs all data subsampling and mini-batch permutations.

---

## 4. End-to-End Reproduction Commands

### Step 1: Verify Dataset Integrity
```powershell
python scripts/audit_dataset_quality.py --dataset-dir data/qism_research_dataset
```
Verifies pyarrow schemas, confirms zero missing values, verifies that `train` has $0\%$ attack pulses, and validates scenario-level isolation.

### Step 2: Execute the Complete Model Benchmark
```powershell
python scripts/run_complete_benchmark.py `
    --dataset-root data/qism_research_dataset `
    --results-root results/qism_model_benchmark `
    --seed 42
```
Trains all 29 models, evaluates validation thresholds, benchmarks across all 3 test partitions, saves checkpoints, outputs all CSV/Parquet tables, and exports publication figures.

### Step 3: Run the Verification Test Suite
```powershell
pytest qml/pitqsvdd/tests/ -v
```
Executes all 36 unit and integration tests verifying circuit constructions, gradient propagation on `lightning.qubit`, loss functions, metric calculations, and checkpoint serialization.

---

## 5. Artifact Verification Checklist

Following execution, confirm the presence and integrity of the following artifact files:

- [x] `results/qism_model_benchmark/checkpoints/classical/` (9 classical checkpoints)
- [x] `results/qism_model_benchmark/checkpoints/baselines/` (4 supervised checkpoints)
- [x] `results/qism_model_benchmark/checkpoints/quantum/` (4 quantum baseline checkpoints)
- [x] `results/qism_model_benchmark/checkpoints/pitqsvdd/` (14 PI-TQSVDD ablation checkpoints)
- [x] `results/qism_model_benchmark/tables/model_inventory.csv`
- [x] `results/qism_model_benchmark/tables/per_run_metrics.csv`
- [x] `results/qism_model_benchmark/tables/aggregate_metrics.csv`
- [x] `results/qism_model_benchmark/tables/per_attack_metrics.csv`
- [x] `results/qism_model_benchmark/tables/pitqsvdd_ablation_table.csv`
- [x] `results/qism_model_benchmark/tables/compute_cost_table.csv`
- [x] `results/qism_model_benchmark/tables/experiment_registry.csv`
- [x] `results/qism_model_benchmark/figures/comparisons/01_overall_auroc_comparison.png`
- [x] `results/qism_model_benchmark/figures/comparisons/02_recall_at_1pct_fpr_comparison.png`
- [x] `results/qism_model_benchmark/figures/comparisons/03_performance_vs_compute_frontier.png`
- [x] `results/qism_model_benchmark/figures/pitqsvdd/04_qubit_count_sweep.png`
- [x] `results/qism_model_benchmark/figures/pitqsvdd/05_encoding_comparison.png`
- [x] `results/qism_model_benchmark/figures/pitqsvdd/06_temporal_ablation_comparison.png`
- [x] `results/qism_model_benchmark/figures/generalization/07_generalization_robustness.png`
