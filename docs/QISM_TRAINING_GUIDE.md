# Quantum Infrastructure Security Monitor (QISM)
## Training and Experiment Execution Guide

This guide provides the exact commands, scripts, workflows, and procedures to train, checkpoint, evaluate, and reproduce all models in the QISM research benchmark.

---

## 1. Environment Verification

Ensure the Python 3.10 virtual environment is active and all required packages are present:

```powershell
# Activate project virtual environment
.\.venv\Scripts\Activate.ps1

# Verify PennyLane and lightning.qubit backend
python -c "import pennylane as qml; dev = qml.device('lightning.qubit', wires=4); print('PennyLane:', qml.__version__, '| Device: lightning.qubit verified!')"

# Verify PyTorch and PyArrow
python -c "import torch, pyarrow; print('PyTorch:', torch.__version__, '| PyArrow:', pyarrow.__version__)"
```

---

## 2. Running the Complete Benchmark Suite

To execute the entire end-to-end benchmark—including all classical models, supervised baselines, quantum baselines, the full PI-TQSVDD ablation grid, metrics computation, and publication figure generation:

```powershell
# Run the complete automated benchmark pipeline
python scripts/run_complete_benchmark.py `
    --dataset-root data/qism_research_dataset `
    --results-root results/qism_model_benchmark `
    --seed 42
```

### Outputs Generated:
- **Model Checkpoints**: Saved in `results/qism_model_benchmark/checkpoints/`
  - `classical/`: Isolation Forest, One-Class SVM, LOF, Mahalanobis, Robust Covariance, Autoencoder, Temporal Autoencoder, Deep SVDD.
  - `baselines/`: Logistic Regression, Random Forest, Small MLP, RBF-SVM.
  - `quantum/`: VQC, Quantum Kernel SVM, Quantum Autoencoder, Static Quantum SVDD.
  - `pitqsvdd/`: All 14 PI-TQSVDD ablation checkpoints.
- **Aggregate Metrics Tables**: Saved in `results/qism_model_benchmark/tables/` (`.csv` and `.parquet`).
- **Publication Figures**: High-resolution figures saved in `results/qism_model_benchmark/figures/`.

---

## 3. Training and Benchmarking Specific Model Subsets

### 3.1 Training Classical Anomaly Detectors
To train and evaluate classical baselines only:
```python
from scripts.run_complete_benchmark import QISMBenchmarkRunner

runner = QISMBenchmarkRunner(dataset_root="data/qism_research_dataset")
runner.load_data()
runner.fit_preprocessors()
runner.run_classical_baselines()
runner.save_all_tables()
```

### 3.2 Training Classical Supervised Baselines
To train Logistic Regression, Random Forest, RBF-SVM, and Small MLP:
```python
runner.run_classical_supervised_baselines()
```

### 3.3 Training Quantum ML Models
To train VQC, Quantum Kernel SVM, Quantum Autoencoder, and Static Quantum SVDD on `lightning.qubit`:
```python
runner.run_quantum_baselines()
```

### 3.4 Executing the PI-TQSVDD Ablation Grid
To run the full PI-TQSVDD multi-qubit and multi-encoding ablation suite:
```python
runner.run_pitqsvdd_ablation_grid()
```

---

## 4. Checkpoint Loading and Inference Reproduction

### 4.1 Loading Classical Checkpoints
```python
import joblib

# Load fitted Isolation Forest
ifo = joblib.load("results/qism_model_benchmark/checkpoints/classical/isolation_forest_full.joblib")
anomaly_scores = -ifo.decision_function(X_test)

# Load Classical Deep SVDD
import torch
from qml.pitqsvdd.evaluation.comparisons import ClassicalDeepSVDD

model = ClassicalDeepSVDD(in_features=76)
ckpt = torch.load("results/qism_model_benchmark/checkpoints/classical/deep_svdd.pt")
model.load_state_dict(ckpt["model"])
centre = ckpt["centre"]
```

### 4.2 Loading Flagship PI-TQSVDD Checkpoints
```python
import torch
from qml.pitqsvdd.config import CircuitConfig, ModelConfig
from qml.pitqsvdd.encodings import get_encoding
from qml.pitqsvdd.circuits.variational import VariationalQuantumCircuit
from qml.pitqsvdd.models.temporal_pitqsvdd import TemporalPITQSVDD

# 1. Rebuild circuit architecture
c_cfg = CircuitConfig(n_qubits=4, n_layers=2, encoding_type="angle", backend="lightning.qubit", diff_method="adjoint")
encoder = get_encoding("angle", n_qubits=4)
circuit = VariationalQuantumCircuit(c_cfg, encoder=encoder, seed=42)

# 2. Build model and load trained state
model = TemporalPITQSVDD(circuit=circuit, config=ModelConfig(model_type="temporal_pitqsvdd"))
ckpt = torch.load("results/qism_model_benchmark/checkpoints/pitqsvdd/PITQSVDD_4Q_Angle_Ref.pt")
model.load_state_dict(ckpt["state_dict"])
model.centre.copy_(ckpt["centre"])
model.eval()

# 3. Compute continuous anomaly score on sliding windows: (B, 3, 4)
with torch.no_grad():
    s_combined, s_normal, s_temporal = model.score(x_windows)
```

---

## 5. Resuming Interrupted Runs

All checkpoints save complete configurations, optimizer states, and centre vectors. If an experiment is interrupted, the runner inspects `results/qism_model_benchmark/checkpoints/` and automatically discovers completed models, allowing targeted resumption of pending configurations without rerunning completed baselines.
