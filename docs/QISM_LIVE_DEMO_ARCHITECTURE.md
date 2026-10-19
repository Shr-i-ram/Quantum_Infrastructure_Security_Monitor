# Quantum Infrastructure Security Monitor (QISM)
## Live Demonstration Web Application Architecture

**Document Version:** 1.0.0  
**Classification:** Research Engineering Technical Reference  
**Scope:** QISM Web Demonstration Console, Simulator Adapter, Model Inference Service, and Persistence Layer  
**Date:** October 19, 2026  

---

## 1. Executive Architecture Summary

The **QISM Live Demonstration Web Application** is a high-performance, browser-based security operations center (SOC) console engineered to demonstrate real-time, anomaly-based eavesdropping detection and mitigation on simulated Quantum Key Distribution (QKD) infrastructure.

The system connects the existing Monte Carlo stochastic decoy-state BB84 event simulator directly to trained quantum machine learning (QML) and classical anomaly-detection pipelines, running online inference on continuous causal telemetry windows.

### High-Level Topology

```
+-----------------------------------------------------------------------------------+
|                           BROWSER DASHBOARD (FRONTEND)                            |
|  - React 19 + TypeScript + Vite SPA                                               |
|  - Real-Time SVG Time-Series Telemetry Charts (QBER, Gains, Sifted Rates)         |
|  - Live Anomaly Score vs. Calibrated Threshold Curve                              |
|  - Interactive Physical Attack Injection Console                                  |
|  - Persistent Security Incident Dossier & Investigation Drawer                    |
|  - Model Lab & Side-by-Side Multi-Model Inference Comparison                      |
|  - 76-Feature Continuous Telemetry Explorer with CSV Export                       |
|  - Phase XI Benchmark Leaderboard & High-Resolution Figure Viewer                 |
+-----------------------------------------------------------------------------------+
          |                                                       ^
          | REST API (HTTP/1.1)                                   | WebSocket (/ws/telemetry)
          v                                                       | 2 Hz Real-Time Push
+-----------------------------------------------------------------------------------+
|                        FASTAPI BACKEND SERVICE (PORT 8000)                        |
|  - Asynchronous Lifecycle Manager                                                 |
|  - CORS & Static SPA Hosting                                                      |
|  - WebSocket Broadcast Connection Manager with RFC 8259 NaN/Infinity Sanitizer    |
+-----------------------------------------------------------------------------------+
       |                        |                         |                    |
       v                        v                         v                    v
+---------------+      +------------------+      +----------------+   +-----------------+
|  SIMULATOR    |      |  RECORDED REPLAY |      |   INFERENCE    |   |  PERSISTENCE    |
|   SERVICE     |      |     SERVICE      |      |    SERVICE     |   |    SERVICE      |
|               |      |                  |      |                |   |                 |
| - Monte Carlo |      | - Streams Phase X|      | - Pre-warmed   |   | - SQLite3 DB    |
|   BB84 Engine |      |   Benchmark Par- |      |   6Q PI-TQSVDD |   |   (Thread-Safe) |
| - Dynamic     |      |   quet Telemetry |      | - Fitted Multi-|   | - Sessions Table|
|   Attack      |      | - Causal Window  |      |   Scaler Pipe- |   | - Incidents     |
|   Injection   |      |   Emission       |      |   line (76 dim)|   |   Table         |
| - Causal      |      | - Preserves Full |      | - Calibrated   |   | - Telemetry     |
|   Physical    |      |   Provenance     |      |   Thresholds   |   |   History       |
|   Intervention|      | - Live Detector  |      | - PennyLane    |   | - Audit Trail   |
| - 2 kHz Pulse |      |   Inference      |      |   Lightning    |   |   Log           |
|   Throughput  |      |   Option         |      |   Backend      |   |                 |
+---------------+      +------------------+      +----------------+   +-----------------+
       |                        |                         |                    |
       +------------------------+-------------------------+--------------------+
                                            |
                                            v
                             +-------------------------------+
                             |       INCIDENT SERVICE        |
                             | - Consecutive Alarm Policy    |
                             | - Dynamic Attack Cooldown     |
                             | - Honest Ground Truth /       |
                             |   Prediction Decoupling       |
                             | - Severity Bracket Mapping    |
                             +-------------------------------+
```

---

## 2. Component Design & Implementation

### 2.1 Backend Architecture (`qism-live-demo/backend`)

The backend is built using **FastAPI** on Python 3.10+, utilizing asynchronous event loops for background simulation tasks and WebSocket broadcasting:

1. **`app/main.py`**:
   - Application entry point with lifespan management (`@asynccontextmanager`).
   - Automatically initializes SQLite database schema (`init_db()`).
   - Pre-warms the flagship 6-qubit PI-TQSVDD model (`PITQSVDD_6Q_Angle`) into memory during server startup.
   - Hosts the compiled Single Page Application (SPA) from `frontend/dist/`.
   - Manages active WebSocket connections (`ConnectionManager`) with automated JSON sanitization (`sanitize_for_json`) ensuring zero non-standard IEEE 754 `NaN`/`Infinity` tokens reach browser JSON parsers.

2. **`app/services/simulator_service.py`**:
   - Interacts directly with the physical simulation engine (`src/qism/simulation/simulator.py`).
   - Dispatches chunks of 2,000 stochastic BB84 optical pulses per window.
   - Couples directly to `qism.telemetry.windows.WindowAggregator` to extract all 130 telemetry parameters (including 76 continuous observable features, QBER, sifted rate, signal gain, decoy gain, and detector imbalance).
   - Supports online causal attack injection (`inject_attack`, `clear_attack`) targeting all 6 physical attack classes:
     - `InterceptResendAttack`: Eavesdropper measures in conjugate bases and resends.
     - `PNSAttack`: Photon-Number-Splitting on multiphoton pulses.
     - `DetectorManipulationAttack`: Continuous-wave optical blinding on Bob's SPADs.
     - `ChannelLossAttack`: Physical attenuation / fiber bending attack.
     - `TimingAttack`: Pulse delay and gate synchronization shift.
     - `SourceManipulationAttack`: Trojan-horse laser intensity intervention.

3. **`app/services/inference_service.py`**:
   - Single source of truth for online QML and classical inference.
   - Loads the authoritative fitted preprocessor pipeline (`results/qism_model_benchmark/preprocessors/fitted_pipeline.joblib`).
   - Maintains an online sliding window buffer of length $L=3$.
   - Dispatches inference requests to PennyLane's `lightning.qubit` simulator with C++ adjoint differentiation.
   - Evaluates predictions against fixed, calibrated decision thresholds at the target 1% False Positive Rate (FPR) baseline ($\tau = 0.0452$ for 6Q PI-TQSVDD).

4. **`app/services/replay_service.py`**:
   - Manages playback of verified research-grade Parquet telemetry generated during the Phase X benchmark.
   - Serves runs from `data/qism_research_dataset/test_in_distribution/shard_0000/attacks/`.
   - Distinguishes **Recorded Scenario Replay** from live simulation with explicit persistent UI badges and provenance tags.
   - Feeds historical telemetry into the active loaded detector model, enabling fair, reproducible evaluations under presentation conditions.

5. **`app/services/incident_service.py`**:
   - Manages operational alerting logic:
     - Minimum consecutive anomalous windows: $k = 1$
     - Alert suppression cooldown: $C = 3$ windows
     - Severity brackets:
       - $\mathcal{A}_t \ge 2.5\tau$: `CRITICAL`
       - $\mathcal{A}_t \ge 1.8\tau$: `HIGH`
       - $\mathcal{A}_t \ge 1.2\tau$: `MEDIUM`
       - $\mathcal{A}_t \ge \tau$: `LOW`
   - Strictly enforces **separation of ground truth from unsupervised detection**: Unsupervised anomaly detectors score deviations from clean link state space; the simulator's injected attack identity is logged separately as ground truth evidence.

6. **`app/database.py`**:
   - Thread-safe SQLite engine (`qism_demo.db`).
   - Persists `sessions`, `incidents`, `telemetry_history`, and `audit_logs`.
   - Survives browser refreshes, reconnects, and server restarts.

---

### 2.2 Frontend Architecture (`qism-live-demo/frontend`)

The frontend is a modular Single Page Application built with **React 19**, **TypeScript**, and **TailwindCSS v4**:

1. **`src/views/LiveMonitor.tsx`**:
   - Primary SOC presentation view.
   - High-contrast visual telemetry strip (Elapsed time, Windows, Anomaly Score, Calibrated Threshold, Latency, QBER).
   - Dynamic SVG Anomaly Score chart with real-time sliding trace, threshold line, alert markers, and ground-truth attack shading.
   - Multi-signal Physical Telemetry chart (QBER, Sifted Key Rate, Decoy-State Gains).
   - Operator control console (Live vs Replay mode, Start, Pause, Resume, Stop, Reset).
   - Physical Attack Injection Console with severity and duration sliders.

2. **`src/views/Incidents.tsx`**:
   - Searchable, filterable incident management table.
   - Severity and status filtering (`NEW`, `ACKNOWLEDGED`, `INVESTIGATING`, `RESOLVED`).
   - Interactive Evidence Dossier Drawer displaying full telemetry snapshots, PCA representations, and detection latencies.

3. **`src/views/TelemetryExplorer.tsx`**:
   - Exhaustive dictionary view of all 76 observable continuous features.
   - Categorized by: QBER, Rates & Counts, Gains & Transmittance, Detector Dynamics, Timing & Polarization, Environmental Sensors.
   - Real-time numerical display, feature trend sparkline, and one-click CSV export.

4. **`src/views/ModelLab.tsx`**:
   - Multi-model catalog displaying 13 trained and verified models.
   - Displays quantum circuit parameters (qubit count, circuit depth, encoding method).
   - Side-by-side comparative inference runner evaluating up to 4 models simultaneously on identical telemetry windows.

5. **`src/views/EvaluationResults.tsx`**:
   - Authoritative benchmark leaderboard loaded from `results/qism_model_benchmark/tables/aggregate_metrics.csv`.
   - Publication figure gallery displaying all 7 high-resolution research figures (AUROC comparisons, compute frontiers, qubit sweeps, encoding ablations, and out-of-distribution stress tests) with full-screen modal zoom.

6. **`src/views/SystemHealth.tsx`**:
   - Real-time diagnostic monitors for backend connectivity, database health, quantum backend readiness, and memory utilization.
   - Pre-demonstration readiness checklist verifying all 8 operational gates before a live presentation.

---

## 3. Communication & Data Flow Protocols

### 3.1 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health, component status, host memory & CPU |
| `GET` | `/api/models` | List all verified models and active model selection |
| `POST` | `/api/models/select` | Activate a specific detector model |
| `POST` | `/api/models/compare` | Run side-by-side inference across multiple models |
| `GET` | `/api/scenarios/attacks` | Catalog of physically supported simulator attacks |
| `GET` | `/api/scenarios/replays` | Catalog of verified recorded benchmark runs |
| `GET` | `/api/sessions/current` | Active session metadata and status |
| `POST` | `/api/sessions/start` | Start live stochastic QKD simulation |
| `POST` | `/api/sessions/replay/start` | Start recorded scenario replay |
| `POST` | `/api/sessions/pause` | Pause active simulation or replay |
| `POST` | `/api/sessions/resume` | Resume paused simulation or replay |
| `POST` | `/api/sessions/stop` | Terminate session |
| `POST` | `/api/sessions/reset` | Reset simulator physics to clean baseline |
| `POST` | `/api/sessions/attacks/inject`| Dynamically inject physical attack |
| `POST` | `/api/sessions/attacks/clear` | Clear active physical attack |
| `GET` | `/api/incidents` | Query persistent incidents from SQLite |
| `PATCH`| `/api/incidents/{id}/status` | Update incident workflow status |
| `GET` | `/api/telemetry/schema` | 76-feature telemetry dictionary |
| `GET` | `/api/benchmark/leaderboard` | Authoritative Phase XI benchmark leaderboard |
| `GET` | `/api/benchmark/figures` | Catalog of publication research figures |
| `GET` | `/api/benchmark/figures/{cat}/{file}` | Serve high-resolution figure image file |

### 3.2 WebSocket Streaming Protocol (`/ws/telemetry`)

Each telemetry window (emitted every 0.5 s during active sessions) transmits an RFC 8259-compliant JSON payload:

```json
{
  "type": "telemetry_update",
  "mode": "live_simulation",
  "session_id": "SIM-1792417326",
  "window_index": 8,
  "timestamp": 1792417330.12,
  "simulated_time_ms": 800.0,
  "qber": 0.2142,
  "sifted_rate": 0.0035,
  "signal_gain": 0.0078,
  "decoy_gain": 0.0019,
  "detector_imbalance": 0.012,
  "anomaly_score": 0.1172,
  "threshold": 0.0452,
  "is_anomaly": true,
  "inference_latency_ms": 4.8,
  "model_id": "PITQSVDD_6Q_Angle",
  "ground_truth_attack_active": true,
  "ground_truth_attack_type": "intercept_resend",
  "display_badge": "SIMULATION ENVIRONMENT",
  "scenario_id": "live_default"
}
```

---

## 4. Scientific Integrity & Separation Rules

The application adheres strictly to the research requirements of the project:
1. **No Fabricated Telemetry**: Every displayed metric originates from the Monte Carlo stochastic engine (`Simulator`) or pre-recorded Phase X Parquet datasets.
2. **No Fabricated Anomaly Scores**: The anomaly score is computed by executing real tensor operations and quantum circuit simulations through PennyLane.
3. **Threshold Integrity**: Decision thresholds are strictly loaded from offline calibration artifacts (`CALIBRATED_THRESHOLDS`) calibrated at target 1% False Positive Rate on normal validation data. Thresholds are never modified dynamically to force an alert.
4. **Honest Evaluation Reporting**: Ground truth attack state (known to the simulator) is strictly separated from the unsupervised model's anomaly score prediction.
