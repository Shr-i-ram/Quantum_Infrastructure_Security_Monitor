# QISM Research Dataset Reproducibility Guide

## 1. Environment Requirements & Setup

The QISM dataset generation pipeline is deterministic and reproducible under Python 3.10.

### Prerequisites
- OS: Windows 10/11 or Linux
- Python: 3.10.11
- Virtual Environment: `.venv/`

### Key Dependencies
```text
numpy >= 2.0.0
scipy >= 1.15.0
pandas >= 2.2.0
pyarrow >= 19.0.0
pydantic >= 2.10.0
pyyaml >= 6.0.0
tqdm >= 4.66.0
matplotlib >= 3.10.0
```

---

## 2. Deterministic Seed Policy

The generation pipeline enforces a strict hierarchical seed derivation tree:
```text
Master Seed (e.g. 20261019)
    │
    ├── derive_dataset_seed(master_seed, split_name)
    │       │
    │       ├── derive_scenario_seed(split_seed, scenario_id)
    │       │       │
    │       │       └── derive_run_seed(scenario_seed, run_id)
```
- Every worker process is initialized with its own independent `job.seed`.
- Simulations run bit-identically whether executed in serial or across $N$ parallel workers.
- Parallel worker race conditions never affect RNG stream sequences.

---

## 3. Exact Commands

### A. Preview Scenarios & Workload Estimation
To inspect the scenario population, parameter coverage, and pulse/window estimates without running simulations:
```bash
python -m qism generate-scenarios configs/qism_research_large.yaml
```

### B. Run Pilot Generation (Stage A Benchmark)
To execute a quick 20-run validation pilot across all 6 partitions:
```bash
python -m qism generate-dataset configs/pilot_research.yaml --workers 4 --fresh
```

### C. Execute Full Production Dataset Generation
To generate the complete 200-scenario, 505-run, 10.1-million-pulse research dataset:
```bash
python -m qism generate-dataset configs/qism_research_large.yaml --workers 4
```

### D. Resume an Interrupted Generation Run
If the process is stopped or interrupted at any point, resume safely without duplicate execution or data loss:
```bash
python -m qism resume configs/qism_research_large.yaml --workers 4
```
Alternatively:
```bash
python -m qism generate-dataset configs/qism_research_large.yaml --workers 4 --resume
```
The scheduler reads `checkpoints/job_manifest.json`, identifies completed jobs, skips them, and resumes pending runs.

### E. Validate Complete Dataset & Check Zero Leakage
To run the automated schema compliance, ground-truth separation, and scenario-split isolation audit:
```bash
# Validate individual split
python scripts/validate_dataset.py data/qism_research_dataset/train

# Validate all splits and verify scenario-level isolation
python scripts/validate_dataset.py \
    data/qism_research_dataset/calibration \
    data/qism_research_dataset/train \
    data/qism_research_dataset/validation \
    data/qism_research_dataset/test_in_distribution \
    data/qism_research_dataset/test_generalization \
    data/qism_research_dataset/test_stress \
    --check-split
```

---

## 4. Verification Checksums & Manifests

Each generated dataset directory contains:
- `dataset_manifest.json`: Root provenance record with configuration hash, Python version, library versions, total scenario/run counts, and wall time.
- `manifest.json`: Standard mirror of root provenance.
- `<partition>/manifest.json`: Per-split manifest detailing completed runs, attack fractions, and edge case parameters.
- `<partition>/<shard>/shard_manifest.json`: Logical shard listing linking runs to physical parquet locations.
- `schema/telemetry_schema.json`: Complete machine-readable field specification.
- `scenario_registry/scenario_registry.parquet`: Machine-readable parameters of every scenario instance.
