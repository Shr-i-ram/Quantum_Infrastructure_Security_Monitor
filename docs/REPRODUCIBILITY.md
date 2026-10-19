# QISM V1.0 Research Prototype Reproducibility Guide

**Project:** Quantum Infrastructure Security Monitor (QISM)  
**Milestone:** V1.0 — Integrated, Validated, Reproducible Research Prototype  
**Date:** October 2026  

---

## 1. System Requirements & Environment Setup

### 1.1 Python Version & Virtual Environment
QISM V1.0 requires **Python 3.10** (consistent with the established project environment).

```bash
# Verify Python version
python --version
# Expected: Python 3.10.x

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies in editable mode
pip install --upgrade pip
pip install -e .
pip install pytest matplotlib seaborn scikit-learn torch pandas
```

---

## 2. Running Verification & Regression Tests

To run the complete test suite across all milestones (V0.1 through V1.0):

```bash
# Run full test suite (245 tests)
python -m pytest

# Run specific V1.0 hardening and reproducibility test suites
python -m pytest tests/test_v10_hardening.py tests/test_v10_reproducibility.py -v
```

Expected result: **245 passed in ~190s (100% pass rate, 0 failures, 0 errors)**.

---

## 3. Canonical End-to-End Demonstration (`qism v10-demo`)

To execute the single-command end-to-end research prototype demonstration:

```bash
# Run canonical V1.0 demo on drift-then-attack scenario
python -m qism.cli v10-demo --scenario drift_then_attack --outdir outputs/v1_0/demo
```

### What this command executes:
1. Generates sequential BB84 telemetry experiencing benign loss drift (+2.2 dB) followed by an Intercept-Resend attack.
2. Ingests and sanitizes records, verifying physical limits and stripping ground truth.
3. Computes causal rolling window deviations and 1-Wasserstein drift distance.
4. Gated adaptation updates reference distribution during drift and locks during attack.
5. Fuses heterogeneous signals into evidence score $F_t \in [0, 1]$.
6. Emits structured alerts and correlates 93 raw alarms into 1 coherent security incident (93.0x compression).
7. Writes an append-only JSONL audit trail, an interactive HTML dashboard, and timeline PNG.

Expected runtime: **~6 to 8 seconds**.

---

## 4. Master Benchmark Suite Reproduction (`qism v10-benchmark`)

To reproduce the complete 12-scenario benchmark matrix across 3 seeds and all 5 ablation studies:

```bash
# Execute master benchmark matrix across seeds 42, 101, and 2024
python -m qism.cli v10-benchmark --outdir outputs/v1_0 --seeds 42,101,2024
```

### Generated Artifacts Location:
- `outputs/v1_0/manifests/v10_benchmark_manifest.json` — Complete benchmark metadata and run index.
- `outputs/v1_0/metrics/v10_benchmark_summary.csv` — Aggregated metrics table reporting mean ± std across seeds.
- `outputs/v1_0/metrics/v10_benchmark_raw_seeds.csv` — Complete raw records for all 36 runs.
- `outputs/v1_0/metrics/v10_ablations.json` — Structured metrics for Comparisons A, B, C, D, E.
- `outputs/v1_0/audit_logs/v10_audit_log.jsonl` — Append-only audit log of all transitions.
- `outputs/v1_0/plots/` — 13 publication-quality PNG charts (resolution 300 DPI).
- `outputs/v1_0/dashboard.html` — Interactive visualization dashboard.

Expected runtime: **~35 to 45 seconds**.

---

## 5. Random Seed Handling & Determinism Invariants

To guarantee exact numerical reproducibility across experiments:
- All scenario generation and model initialization use the global seeding helper `qism.random.seed_all(seed)`:
  ```python
  from qism.random import seed_all
  seed_all(42)
  ```
- This helper synchronizes Python `random`, NumPy `np.random`, and PyTorch CPU PRNG states.
- Running `python -m pytest tests/test_v10_reproducibility.py` validates that repeated runs with identical seeds yield bit-for-bit identical telemetry, incident boundaries, and evidence scores.

---

## 6. Known Reproducibility Limitations

1. **Operating System & Floating-Point Differences:** Minor floating-point discrepancies ($\le 10^{-12}$) may occur across different CPU microarchitectures (e.g. x86_64 FMA vs ARM NEON). All assertions in tests use `np.testing.assert_array_almost_equal` with standard machine tolerances.
2. **GPU Non-Determinism Avoided:** QISM runs on PyTorch CPU tensors to eliminate non-deterministic atomic operations present in CUDA backends.
3. **Simulation Boundary:** Reproducing experiments validates detection within the BB84 simulation model; it does not replace empirical data collection on physical quantum hardware.
