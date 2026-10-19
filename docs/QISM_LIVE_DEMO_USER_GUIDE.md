# Quantum Infrastructure Security Monitor (QISM)
## Live Demonstration User Guide & Presentation Manual

**Document Version:** 1.0.0  
**Target Audience:** Final Review Panel, Research Evaluators, Technical Operators  
**System:** QISM Live Demonstration SOC Web Console  
**Date:** October 19, 2026  

---

## 1. Quick Start Instructions

The QISM Live Demonstration Web Application is completely self-contained and runs locally without requiring an internet connection.

### Prerequisites

- **Python:** 3.10+ in the repository `.venv`
- **Node.js:** v20+ (for building frontend assets, if modifying code)
- **Host:** Windows, macOS, or Linux

### Starting the Demonstration Console

To launch the integrated console (FastAPI backend + pre-built React frontend):

```powershell
# From the repository root (qism-simulator):
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir qism-live-demo/backend --host 127.0.0.1 --port 8000
```

Once launched, open your web browser and navigate to:

👉 **`http://127.0.0.1:8000/`**

*(Note: If you wish to run the Vite frontend development server concurrently for hot module reloading, run `npm run dev` inside `qism-live-demo/frontend/` and access `http://localhost:5173/`).*

---

## 2. Navigating the Console

The interface is structured into six functional security operations views accessible via the left navigation sidebar:

1. **Live Monitor (`/`)**:
   - The primary demonstration screen. Shows real-time sliding window telemetry, Anomaly Score vs Decision Threshold curves, physical simulator controls, and the dynamic attack injection console.
2. **Incidents (`/incidents`)**:
   - Searchable, filterable incident management registry. Displays all alerts logged to SQLite, allowing inspection of complete JSON evidence snapshots and workflow status tracking (`ACKNOWLEDGED`, `INVESTIGATING`, `RESOLVED`).
3. **Telemetry Explorer (`/telemetry`)**:
   - Interactive catalog of all 76 continuous observable features across QBER, Gains, Detector Dynamics, Timing, and Environmental sensors.
4. **Model Lab (`/models`)**:
   - Catalog of 13 verified trained checkpoints. Default flagship model is **6-qubit PI-TQSVDD**. Run live side-by-side comparative inference across classical, quantum, and baseline models on identical telemetry windows.
5. **Evaluation Results (`/evaluation`)**:
   - Authoritative benchmark leaderboard from the Phase XI evaluation and an interactive gallery of the 7 publication-grade high-resolution figures.
6. **System Health (`/health`)**:
   - Diagnostics for the PennyLane quantum backend, SQLite persistence, fitted preprocessor pipelines, and an 8-item pre-demonstration readiness checklist.

---

## 3. The 5–7 Minute Live Demonstration Script

Follow this step-by-step walkthrough to present the research project to an evaluation panel.

### Step 1: System Readiness & Context (1 Minute)
1. Open the console at `http://127.0.0.1:8000/`.
2. Direct the panel's attention to the header:
   - Point out the **QISM v1.0 Demo Console** branding and the active detector: **PI-TQSVDD (6 Qubits)**.
   - Note the **System Standby** badge and **Live Stream 2Hz** status.
3. Click **System Health** in the sidebar:
   - Walk through the **Pre-Demonstration Readiness Checklist**: Show that the PennyLane quantum backend, SQLite database, fitted 76-feature pipeline, and 6-qubit PI-TQSVDD checkpoint are all loaded and verified.

### Step 2: Live Baseline Monitoring (1.5 Minutes)
1. Return to **Live Monitor** in the sidebar.
2. Under the **Simulation Controller** card on the right, ensure **Live Simulator** is selected.
3. Click the blue **Start Session** button.
4. Observe the console transitioning to active operation:
   - The header displays the glowing **SIMULATION ENVIRONMENT** badge.
   - Telemetry windows increment at 2 Hz (2,000 pulses simulated per window).
   - In the **Detector Anomaly Score vs Decision Threshold** chart, point out the blue anomaly score trace staying well below the red dashed threshold line ($\tau = 0.0452$).
   - In the **Physical QKD Channel Telemetry** chart, point out the stable baseline: QBER is minimal ($< 1.0\%$), and the sifted key rate is steady at $\sim 3.5$ kEv/s.
   - Explain to the panel: *"Under clean environmental conditions, the 6-qubit PI-TQSVDD model maps the optical pulse sequence to a compact hypersphere in Hilbert space, producing low anomaly scores."*

### Step 3: Physical Attack Injection & Detection (2 Minutes)
1. Direct the panel's attention to the **Physical Attack Injection** panel on the right.
2. Explain the causal attack model:
   - Under **Attack Mechanism**, select **Intercept-Resend Attack** (Eve measures in conjugate bases and resends).
   - Point out the severity slider ($80\%$) and duration ($15$ windows).
   - Emphasize to the panel: *"We do not fake alerts or directly modify telemetry numbers. The attack engine injects physical parameters into the Monte Carlo quantum optics pipeline."*
3. Click the red **Inject Attack** button:
   - The header immediately flags **ATTACK ACTIVE: INTERCEPT RESEND**.
   - In the bottom right **Scientific Ground Truth Verification** card, observe that the simulator reports Ground Truth as **INTERCEPT_RESEND**.
4. Watch the detector respond in real time:
   - QBER rapidly spikes from $0.0\%$ to $\sim 21.4\%$.
   - The 6-qubit PI-TQSVDD anomaly score jumps from $0.015$ to over $0.110$, crossing the calibrated threshold ($\tau = 0.0452$).
   - The high-visibility red **SECURITY INCIDENT DETECTED** banner appears at the top.
   - The detector status card confirms: **TRUE POSITIVE (Detection OK)**.
5. Click **Stop** on the Simulation Controller.

### Step 4: Incident Investigation & Evidence Dossier (1.5 Minutes)
1. Click **Incidents** in the sidebar.
2. In the incident table, locate the newly logged incident (e.g. `INC-SIM-1792-0008`):
   - Point out that it recorded the exact timestamp, severity (`CRITICAL`), anomaly score ($0.1172$), threshold ($0.0452$), and detection latency ($1$ window).
3. Click the **Inspect Evidence** eye icon to open the **Evidence Dossier**:
   - Show the panel the recorded physical snapshot: QBER, Signal Gain, Decoy Gain, and the model's internal tensor decomposition.
   - Click the **ACKNOWLEDGED** workflow button: Observe the status change to `ACKNOWLEDGED` and explain how SOC analysts track incident lifecycles.
   - Emphasize: *"All incidents are persisted in SQLite and survive page reloads and browser disconnects."*

### Step 5: Multi-Model Comparison & Research Benchmark (1 Minute)
1. Click **Model Lab** in the sidebar:
   - Show the 13 available models in the catalog (PI-TQSVDD 6Q, 4Q Reference, 4Q Amplitude, 4Q Data-Reuploading, Classical Temporal GRU Autoencoder, Isolation Forest, etc.).
   - Click **Run Multi-Model Comparison**: Show the panel the side-by-side inference outputs on the identical telemetry window, demonstrating that PI-TQSVDD achieves superior sensitivity compared to static classical baselines.
2. Click **Evaluation Results** in the sidebar:
   - Point out the **Benchmark Leaderboard** with test AUROCs, AUPRCs, and recall at 1% FPR.
   - Click the **Publication Figures** tab and click **Figure 1** to expand the high-resolution publication comparison chart.
3. Conclude the demonstration:
   - Return to **Live Monitor** to confirm the console is in a clean standby state.

---

## 4. Recorded Scenario Replay Mode (Reliability Fallback)

If a demonstration computer experiences high CPU load or you wish to show a deterministic, pre-recorded research scenario:

1. On **Live Monitor**, toggle the switch from **Live Simulator** to **Replay Mode**.
2. Select a verified scenario from the dropdown:
   - **Photon Number Splitting (PNS) Attack Scenario** (`run_0000`)
   - **Intercept-Resend Attack Scenario** (`run_0003`)
   - **Channel Loss Attack Scenario** (`run_0001`)
   - **Normal Stable Baseline** (`run_0000`)
3. Select playback speed ($0.5\times$, $1.0\times$, or $2.0\times$).
4. Click **Start Session**:
   - The header displays the amber **RECORDED SCENARIO REPLAY** badge.
   - Telemetry windows stream from verified Phase X Parquet files while the loaded 6-qubit PI-TQSVDD model performs real online inference.

---

## 5. Troubleshooting & Diagnostics

| Symptom | Cause | Resolution |
|---|---|---|
| Red *Disconnected* indicator in header | Backend Uvicorn process is not running on port 8000 | Launch backend with `.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir qism-live-demo/backend --host 127.0.0.1 --port 8000` |
| Port 8000 already in use | Another process is bound to port 8000 | In PowerShell: `Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess \| Stop-Process` |
| Checkpoint `MISSING` in Model Lab | Checkpoint path not found | Verify checkpoint exists in `results/qism_model_benchmark/checkpoints/pitqsvdd/PITQSVDD_6Q_Angle.pt` |
| Attack injected but no alert | Attack severity too low or duration expired | Use severity $\ge 0.70$ and duration $\ge 10$ windows for immediate threshold crossing |
| Database reset needed | Operator wants to clear demo history | Delete `qism-live-demo/qism_demo.db`; the backend will auto-recreate a fresh schema on startup |
