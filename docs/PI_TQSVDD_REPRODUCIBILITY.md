# QISM PI-TQSVDD — Reproducibility and Execution Guide
## Complete Instructions for Verification, Benchmark Sweeps, Resumption, and Final Training

---

### Executive Reproducibility Metadata
- **Project:** Quantum Infrastructure Security Monitor (QISM)
- **Document:** Reproducibility Guide (`docs/PI_TQSVDD_REPRODUCIBILITY.md`)
- **Python Environment:** `.venv` (Python 3.10.11 AMD64 Windows)
- **Primary Simulator Backend:** `lightning.qubit` (PennyLane 0.42.3 + `pennylane-lightning` 0.42.0)
- **Audit Date:** Current Baseline (October 2026)

---

## 1. Environment Verification

Before running any script, ensure the isolated virtual environment is active and the required quantum backend is operational:

```powershell
# Set PYTHONPATH to project root
$env:PYTHONPATH="."

# Verify Python and PennyLane lightning.qubit installation
& ".\.venv\Scripts\python.exe" -c "import pennylane as qml, torch; print('PennyLane:', qml.__version__); dev = qml.device('lightning.qubit', wires=4); print('Backend Verified:', dev.name)"
```

Expected output:
```text
PennyLane: 0.42.3
Backend Verified: lightning.qubit
```

---

## 2. Automated Test Suite (Validation Baseline)

To run the complete automated test suite (36 unit and integration tests):

```powershell
$env:PYTHONPATH="."
& ".\.venv\Scripts\pytest.exe" -v qml/pitqsvdd/tests
```

**Test Coverage Verified:**
- `test_checkpoint_resume.py`: Validates crash-resistant atomic serialization, process restart, parameter restoration, and step continuation.
- `test_circuits.py`: Validates forward passes, finite outputs, and adjoint backward passes across **2, 4, 6, and 8 qubits** on `lightning.qubit`.
- `test_data.py`: Validates zero attack contamination in training sets, causal sliding windows, and scenario boundary isolation.
- `test_encodings.py`: Validates all 6 encodings (Angle, Data Reuploading, Amplitude, IQP/ZZ, Feature Group, Learned Projection).
- `test_losses.py`: Validates SVDD normality loss, temporal residual loss, and multi-centre loss.
- `test_metrics.py`: Validates point-level metrics, threshold calibration, and incident-level delay metrics.

---

## 3. Execution Modes

### Mode A — Smoke Tests (Fast Verification)
To execute a quick, single-pass training run (2 epochs, 2 qubits) verifying device setup, forward pass, gradients, and checkpoint writes:

```powershell
$env:PYTHONPATH="."
& ".\.venv\Scripts\python.exe" -m qml.pitqsvdd.train `
    --config qml/pitqsvdd/configs/smoke.yaml `
    --run-id smoke_test_cli `
    --resume auto
```
Expected runtime: ~2.5 seconds. Outputs logged to `artifacts/pi_tqsvdd/metrics/smoke_test_cli.json`.

---

### Mode B — Pilot Benchmark Sweep (Architecture Investigation)
To execute the multi-configuration benchmark suite across 2, 4, 6, 8 qubits, all 6 encodings, classical baselines, and figure generation:

```powershell
$env:PYTHONPATH="."
& ".\.venv\Scripts\python.exe" qml/pitqsvdd/benchmark_runner.py
```
Outputs generated:
- Registry: `artifacts/pi_tqsvdd/experiment_registry.csv` and `.json`
- Manifests: `artifacts/pi_tqsvdd/manifests/*.json`
- Metrics: `artifacts/pi_tqsvdd/metrics/*.json`
- Checkpoints: `artifacts/pi_tqsvdd/checkpoints/*/checkpoint_latest.pt`
- Figures: `artifacts/pi_tqsvdd/figures/*.png`

---

### Mode C — Definitive Final Full Training Run (Ready for User Execution)
The final training configuration is prepared in `qml/pitqsvdd/configs/pitqsvdd_final.yaml`.

**To execute the definitive final training run:**

```powershell
$env:PYTHONPATH="."
& ".\.venv\Scripts\python.exe" -m qml.pitqsvdd.train `
    --config qml/pitqsvdd/configs/pitqsvdd_final.yaml `
    --run-id pitqsvdd_final `
    --resume auto
```

#### Final Run Characteristics & Computational Estimates
Based on pilot measurements:
- **Architecture:** Temporal PI-TQSVDD with Data Reuploading ($L=2$, $K=2$ reupload blocks)
- **Qubit Count:** 4 qubits on `lightning.qubit` with adjoint differentiation
- **Feature Pipeline:** 48 observable features $\to$ 16 High-Dimensional PCA components $\to$ 4-qubit multi-layer reuploading
- **Training Epochs:** 30 maximum epochs with early-stopping patience of 8 epochs
- **Estimated Runtime:** ~8 to 12 seconds total on CPU
- **Estimated Memory Footprint:** < 45 MB RAM
- **Checkpoint Footprint:** ~150 KB per checkpoint file
- **Resumability:** If interrupted at any epoch, re-running the exact same command with `--resume auto` continues seamlessly from the latest saved checkpoint.

---

## 4. Checkpoint Resumption Commands

### Auto-Resume Latest Checkpoint
```powershell
$env:PYTHONPATH="."
& ".\.venv\Scripts\python.exe" -m qml.pitqsvdd.train `
    --config qml/pitqsvdd/configs/pitqsvdd_final.yaml `
    --run-id pitqsvdd_final `
    --resume auto
```

### Force Fresh Restart (Refusing Overwrite)
```powershell
$env:PYTHONPATH="."
& ".\.venv\Scripts\python.exe" -m qml.pitqsvdd.train `
    --config qml/pitqsvdd/configs/pitqsvdd_final.yaml `
    --run-id pitqsvdd_final `
    --resume none
```
*(Will raise an error if an existing run is found, preventing accidental loss of data).*

### Force Fresh Restart (Explicit Overwrite)
```powershell
$env:PYTHONPATH="."
& ".\.venv\Scripts\python.exe" -m qml.pitqsvdd.train `
    --config qml/pitqsvdd/configs/pitqsvdd_final.yaml `
    --run-id pitqsvdd_final `
    --resume none `
    --overwrite
```

---

## 5. Known Reproducibility Limitations
1. **PyTorch Weights-Only Default:** PyTorch 2.6+ defaults to `weights_only=True` in `torch.load`. The PI-TQSVDD checkpoint manager explicitly sets `weights_only=False` to allow restoring complete NumPy and Python random-number generator state dictionaries.
2. **Determinism:** Bitwise deterministic continuation is achieved on CPU by restoring PyTorch, NumPy, and Python RNG states from checkpoints.
