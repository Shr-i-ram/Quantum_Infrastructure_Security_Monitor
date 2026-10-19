# Quantum Infrastructure Security Monitor (QISM) — Live Demonstration Web Application

An interactive, production-grade web application that demonstrates real-time Quantum Key Distribution (QKD) physical simulation, dynamic attack injection, live streaming anomaly detection across quantum and classical models, and operational incident management.

---

## Features

- **Live Physical Telemetry Streaming**: Real-time simulation of decoy-state BB84 protocol parameters (QBER, raw detection counts, sifted count, dark count rates, optical visibility, phase error, detector temperatures, bias voltages) streamed over WebSockets.
- **Dynamic Attack Injection**: Dynamically inject physical attack vectors on-the-fly:
  - Photon Number Splitting (PNS)
  - Detector Blinding & Control
  - Afterpulse Manipulation
  - Intercept-Resend
  - Channel Attenuation Drift
  - Timing Jitter Shift
- **Multi-Model Streaming Inference**: Real-time scoring using verified model checkpoints from `results/qism_model_benchmark/`:
  - Flagship: **PI-TQSVDD** (Physics-Informed Temporal Quantum Support Vector Data Description)
  - Quantum: Variational Quantum Classifier (VQC), Quantum Autoencoder (QAE), Quantum Kernel SVM
  - Classical Deep: Temporal Autoencoder, Dense Autoencoder, Deep SVDD
  - Classical Statistical: Isolation Forest, Local Outlier Factor (LOF), Mahalanobis Distance, Robust Covariance
- **Incident Lifecycle Management**: Correlates raw alarms into actionable security incidents (`OPENING -> ACTIVE -> ESCALATED -> RESOLVED`) with advisory mitigations and non-executing operator playbooks.
- **Evidence Dossier & Export**: Detailed investigation drawer with per-feature z-scores, telemetry timelines, and exportable forensic dossiers (JSON/CSV).
- **Benchmark & Research Viewer**: Interactive visualization of benchmark tables, ROC curves, PR curves, and ablation matrices.

---

## Architecture

```
qism-live-demo/
├── backend/                  # FastAPI asynchronous backend
│   ├── app/
│   │   ├── api/              # REST endpoints (health, simulation, models, incidents, telemetry)
│   │   ├── services/         # Simulation runner, inference service, incident engine
│   │   ├── config.py         # App configuration & paths
│   │   ├── database.py       # SQLite database initialization
│   │   ├── main.py           # FastAPI application entrypoint
│   │   └── models.py         # Pydantic schemas & data models
│   └── requirements.txt      # Backend Python dependencies
└── frontend/                 # React + TypeScript + Vite frontend
    ├── src/
    │   ├── components/       # Header, navigation, charts, gauges, modals
    │   ├── views/            # Dashboard, incidents, telemetry explorer, benchmark view
    │   ├── services/         # API client & WebSocket manager
    │   ├── types/            # TypeScript interfaces
    │   ├── App.tsx           # Main application view container
    │   └── main.tsx          # React application entrypoint
    ├── package.json
    └── vite.config.ts
```

---

## Quickstart

### Prerequisites
- Python 3.10+ (with virtual environment configured)
- Node.js 18+ and npm

### 1. Launch Backend Server
From the repository root (`qism-simulator`):

```bash
# Using existing project virtual environment
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir qism-live-demo/backend --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`. Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Launch Frontend UI
From the `qism-live-demo/frontend` directory:

```bash
cd qism-live-demo/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your web browser.

---

## Operating Instructions

1. **Start Simulation**: Click the green **Start Simulation** button in the header. Telemetry graphs will begin streaming live data points.
2. **Inject Attack**: Open the **Attack Injection** panel, choose an attack type (e.g. *Detector Blinding* or *Photon Number Splitting*), set the severity and duration, and click **Inject Attack**.
3. **Monitor Anomaly Score**: Observe the anomaly score gauge spike above the detection threshold and trigger an incident alert.
4. **Investigate Incident**: Click on the active incident in the alerts feed to review anomalous physical features, root cause hypotheses, and advisory mitigation steps.
5. **Switch Models**: Use the **Model Selector** in the status bar to compare anomaly response across PI-TQSVDD, Temporal Autoencoder, or Isolation Forest.
