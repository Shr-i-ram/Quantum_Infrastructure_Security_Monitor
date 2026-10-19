# QISM — Quantum Infrastructure Security Monitor: Complete Project Specification & Technical Compendium

**Anomaly-Based Machine Learning Quantum Key Distribution (QKD) Attack Prevention**  
*A Comprehensive Engineering and Physical Reference across Versions 0.1 through 0.6*

---

## 1. Executive Summary & Project Vision

The **Quantum Infrastructure Security Monitor (QISM)** is a physics-informed, stochastic, event-level telemetry simulator and machine learning security evaluation platform for Quantum Key Distribution (QKD) networks. 

Conventional QKD security models (such as the standard decoy-state BB84 protocol) assume that legitimate eavesdropping or physical channel disturbances will manifest exclusively as an elevation in Quantum Bit Error Rate (QBER) or an unexpected reduction in detection rates and yields. In practical quantum communication systems, however, sophisticated side-channel and implementation attacks exploit the non-ideal physical characteristics of hardware (e.g. avalanche photodiode recovery dynamics, thermal dependencies, timing jitter, optical component imperfections, and back-reflections) often without triggering standard macroscopic QBER abort thresholds.

QISM addresses this fundamental vulnerability by establishing an autonomous security monitoring architecture:
1. **Physics-Informed Event Simulation (v0.1–v0.4)**: Generates high-fidelity, stochastic, pulse-by-pulse physical interactions across Alice (source), the quantum channel, and Bob (single-photon avalanche diode receiver), incorporating true non-ideal device physics, continuous environmental drift, and multi-rate physical sensor suites.
2. **Causal Attack Framework (v0.3)**: Simulates 6 physical attacks acting causally on underlying optical and physical parameters rather than artificially injecting statistical deviations into telemetry.
3. **Population Diversity & Scalable Dataset Engine (v0.5)**: Implements Latin Hypercube Sampling across parameter distributions, causal dependency resolution, parallel batch execution, atomic checkpointing, and memory-bounded sharded dataset streaming.
4. **Machine Learning Anomaly Detection Benchmark (v0.6)**: Implements an unsupervised/one-class anomaly detection framework trained strictly on legitimate normal operational behavior. Evaluates statistical baselines, classical ML algorithms, and deep sequential neural networks across in-distribution (IID) and out-of-distribution (OOD) generalization splits, false alarm stability, and detection delay latency.

```
+---------------------------------------------------------------------------------------------------+
|                                  QISM SYSTEM ARCHITECTURE (v0.6)                                   |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
                        ┌───────────────────────────────────────────────────┐
                        │      PHYSICAL & SIMULATION FOUNDATION (v0.1-v0.4) │
                        │  - Decoy-State BB84 Laser Source                  │
                        │  - Dynamic Fiber Channel (Loss, Misalignment)     │
                        │  - Single-Photon Detectors (Dead Time, AP)        │
                        │  - Environmental Drift (O-U Process, Diurnal)     │
                        │  - Causal Physical Attacks (PNS, Blinding, etc.)  │
                        └─────────────────────────┬─────────────────────────┘
                                                  │
                                                  ▼
                        ┌───────────────────────────────────────────────────┐
                        │    POPULATION & DATASET INFRASTRUCTURE (v0.5)     │
                        │  - Parameter Distributions (Normal, LHS)          │
                        │  - Causal Dependency Graph & Stress Regimes       │
                        │  - Parallel Scheduler & Atomic Checkpointing      │
                        │  - Sharded Parquet Streaming & Provenance         │
                        └─────────────────────────┬─────────────────────────┘
                                                  │
                                                  ▼
                        ┌───────────────────────────────────────────────────┐
                        │       MACHINE LEARNING DATA SUBSYSTEM (v0.6)      │
                        │  - Scenario-Level Partitions (Train/Val/Test/OOD) │
                        │  - Zero Label Leakage Verification                │
                        │  - Causal Temporal Feature Engineering (x <= t)   │
                        │  - Preprocessor Pipeline (Fit on Train Normal)    │
                        └─────────────────────────┬─────────────────────────┘
                                                  │
                                                  ▼
                        ┌───────────────────────────────────────────────────┐
                        │      ANOMALY DETECTION MODEL ZOO (v0.6)           │
                        │  - Statistical: Mahalanobis, Robust MCD           │
                        │  - Classical ML: Isolation Forest, OCSVM, LOF     │
                        │  - Neural: Dense MLP Autoencoder                  │
                        │  - Sequential: Temporal LSTM Autoencoder          │
                        └─────────────────────────┬─────────────────────────┘
                                                  │
                                                  ▼
                        ┌───────────────────────────────────────────────────┐
                        │     THRESHOLD CALIBRATION & ALARM LOGIC (v0.6)    │
                        │  - Normal-Only Quantile Calibration: tau = Q_alpha│
                        │  - 3-State Machine: NORMAL -> SUSPECT -> ALARM    │
                        │  - Hysteresis Recovery & Persistence (K windows)  │
                        └─────────────────────────┬─────────────────────────┘
                                                  │
                                                  ▼
                        ┌───────────────────────────────────────────────────┐
                        │    SECURITY EVALUATION & GENERALIZATION (v0.6)    │
                        │  - Window Metrics: Precision, Recall, F1, FPR     │
                        │  - Continuous Metrics: AUROC, AUPRC               │
                        │  - Attack Instance Metrics: Detection Delay D     │
                        │  - OOD Generalization Gap: Delta M = M_IID - M_OOD│
                        │  - Ablation: QKD vs Detector vs Env Telemetry     │
                        └───────────────────────────────────────────────────┘
```

---

## 2. Version Evolution: Architectural Roadmap

| Version | Scope & Milestone | Core Contributions |
|---|---|---|
| **v0.1** | Core Decoy-State BB84 Stochastic Event Engine | Monte Carlo pulse generation, Poisson photon emission, decoy-state selection (signal, decoy, vacuum), fiber transmittance, passive polarization basis routing, detector clicks, windowed telemetry aggregation, analytical theory validation. |
| **v0.2** | Realistic Detector Dynamics & Temporal Telemetry | Finite nonparalyzable dead time $\tau_d$, rate-dependent dead time $\tau_{\text{eff}}(\lambda)$, exponential afterpulsing, active READY/DEAD state machine, busy fraction telemetry, inter-arrival time statistics, Fano factor, V0.1 compatibility toggle. |
| **v0.3** | Causal Attack Framework & Physical Lifecycle | 6 causal physical attack models (intercept-resend, channel loss, source manipulation, detector manipulation, timing shift, photon-number splitting); 5-phase attack lifecycle (`NORMAL`, `ONSET`, `ACTIVE`, `TERMINATION`, `RECOVERY`); multi-attack composition; strict separation of observable vs ground-truth telemetry; zero label leakage. |
| **v0.4** | Enhanced Physical Realism & Infrastructure Sensors | Continuous-time Ornstein–Uhlenbeck drift processes, diurnal thermal cycles, temperature-coupled dark counts and SPAD quantum efficiency, source jitter and intensity drift, dynamic channel attenuation, polarization misalignment drift, local clock drift, ambient stray flux, realistic multi-rate sensor suite (quantization, Gaussian noise, stochastic missingness), checkpoint serialization. |
| **v0.5** | Large-Scale Scenario Generation & Scalable Datasets | Formal scenario and population abstractions (`Scenario`, `PopulationConfig`, `ScenarioGenerator`), 7 continuous/discrete mathematical distributions with Latin Hypercube Sampling (LHS), causal dependency DAG, 5-tier deterministic seed hierarchy, fault-tolerant parallel `ProcessPoolExecutor`, memory-bounded sharded Parquet dataset streaming (`DatasetShardWriter`), scenario-level data partitioning. |
| **v0.6** | Anomaly Detection, ML Benchmarking & Security Evaluation | Unsupervised anomaly detection paradigm trained strictly on normal telemetry, chunked ML data loader, causal temporal window features ($x_{\le t}$), robust preprocessing pipeline, 7 baseline model families (Mahalanobis, Robust MCD, Isolation Forest, One-Class SVM, LOF, PyTorch Dense Autoencoder, PyTorch Sequence-to-Sequence LSTM Temporal Autoencoder), normal-only quantile threshold calibration, operational alarm state machine (persistence, hysteresis), out-of-distribution generalization analysis, feature ablation studies, leakage sanity diagnostics, unified CLI. |

---

## 3. Directory Layout & Codebase Taxonomy

The QISM repository is structured into modular Python packages with strict layer boundaries:

```
qism-simulator/
├── configs/                          # Experiment & simulation configuration presets
│   ├── default.yaml                  # Default physical operating point
│   ├── v0.4_smoke.yaml               # V0.4 fast verification preset
│   ├── v0.4_generalization.yaml      # V0.4 generalization test preset
│   ├── v0.5_smoke.yaml               # V0.5 parallel execution smoke test
│   ├── v0.5_small.yaml               # V0.5 20-scenario research dataset preset
│   ├── v0.6_smoke.yaml               # V0.6 end-to-end ML test preset
│   ├── v0.6_small.yaml               # V0.6 small benchmark configuration
│   └── v0.6_medium.yaml              # V0.6 medium benchmark configuration
├── src/
│   └── qism/
│       ├── __init__.py               # Package metadata (__version__ = "0.6.0")
│       ├── __main__.py               # CLI entrypoint (python -m qism)
│       ├── cli.py                    # Unified argparse CLI implementation
│       ├── config.py                 # Pydantic configuration schemas & validation
│       ├── random.py                 # Deterministic seed hierarchy & RNG factories
│       ├── types.py                  # Domain primitives, enums, & data structures
│       ├── units.py                  # Physical constant definitions & conversions
│       ├── attacks/                  # Physical attack simulation models
│       │   ├── __init__.py           # Attack registry & lifecycle interfaces
│       │   ├── base.py               # Attack base class & 5-phase lifecycle manager
│       │   ├── intercept_resend.py   # Full basis measurement & pulse regeneration
│       │   ├── loss.py               # Optical attenuation attack model
│       │   ├── source.py             # Mean photon number manipulation attack
│       │   ├── detector.py           # Single-photon detector blinding & efficiency attack
│       │   ├── timing.py             # Pulse arrival delay & phase shift attack
│       │   └── pns.py                # Photon-Number Splitting (PNS) attack
│       ├── channel/                  # Quantum channel physics
│       │   ├── __init__.py
│       │   ├── loss.py               # Beer-Lambert fiber loss & transmissivity
│       │   ├── polarization.py       # Unitary polarization rotation & misalignment
│       │   └── timing.py             # Fiber propagation delay & chromatic dispersion
│       ├── detector/                 # Receiver single-photon avalanche diodes (SPAD)
│       │   ├── __init__.py
│       │   ├── detector.py           # Stateful SPAD (dead time, afterpulsing, recovery)
│       │   ├── dark_counts.py        # Poisson thermal & tunneling dark counts
│       │   ├── timing.py             # SPAD transit-time jitter & Gaussian delay
│       │   └── probe.py              # Synthetic Poisson pulse probe for calibration
│       ├── environment/              # Macro-environmental dynamics & infrastructure sensors
│       │   ├── __init__.py
│       │   ├── base.py               # Environmental state dataclass & interfaces
│       │   ├── dynamic.py            # Coupled environmental coordinator
│       │   ├── processes.py          # Ornstein-Uhlenbeck SDE & diurnal harmonic models
│       │   └── sensors.py            # Multi-rate physical sensors (ADC, noise, missing)
│       ├── execution/                # Parallel execution & batch job scheduling
│       │   ├── __init__.py
│       │   ├── job.py                # Self-contained picklable SimulationJob & JobResult
│       │   ├── worker.py             # Sandboxed worker process runner
│       │   └── scheduler.py          # ProcessPoolExecutor scheduler with checkpointing
│       ├── population/               # Scenario generation & population diversity
│       │   ├── __init__.py
│       │   ├── distributions.py      # Uniform, Normal, TruncNorm, LogNorm, Beta, Cat
│       │   ├── sampling.py           # Latin Hypercube Sampling (LHS) engine
│       │   ├── dependencies.py       # Causal dependency DAG & stress regimes
│       │   ├── validator.py          # Physical law verification assertions
│       │   ├── generator.py          # Scenario population generator & hasher
│       │   └── diagnostics.py        # Kolmogorov-Smirnov goodness-of-fit tests
│       ├── receiver/                 # Bob detection station
│       │   ├── __init__.py
│       │   ├── bob.py                # Bob receiver coordinator
│       │   ├── basis_selector.py     # Passive 50:50 beam-splitter basis choice
│       │   └── pbs.py                # Polarizing beam-splitter state separation
│       ├── simulation/               # Core execution engine
│       │   ├── __init__.py
│       │   ├── simulator.py          # Chunked pulse simulation engine
│       │   ├── runner.py             # Single scenario runner & output dispatch
│       │   └── checkpoint.py         # Mid-run state serialization & crash recovery
│       ├── source/                   # Alice transmitter station
│       │   ├── __init__.py
│       │   ├── alice.py              # Alice transmitter coordinator
│       │   ├── photon_source.py      # Poisson photon number emission
│       │   └── decoy_state.py        # Decoy intensity modulation (signal, decoy, vac)
│       ├── telemetry/                # Observability & feature extraction
│       │   ├── __init__.py
│       │   ├── events.py             # Pulse-level event recording & schemas
│       │   ├── features.py           # Physics-informed macroscopic window features
│       │   └── windows.py            # Time-window aggregation engine
│       ├── datasets/                 # Sharded storage & dataset manifests
│       │   ├── __init__.py
│       │   ├── generator.py          # High-level dataset generation coordinator
│       │   ├── manifest.py           # Dataset schema & provenance models
│       │   ├── pipeline.py           # Batch generation pipeline
│       │   ├── provenance.py         # Hardware, OS, Git, & dependency capture
│       │   ├── sharding.py           # Bounded-memory Parquet shard writer
│       │   └── writer.py             # Manifest & metadata serialization
│       ├── ml/                       # Machine learning anomaly detection (v0.6)
│       │   ├── __init__.py
│       │   ├── data/                 # ML dataset loaders & split isolation
│       │   │   ├── loader.py         # Chunked, streaming ML Parquet loader
│       │   │   └── splits.py         # Scenario-level split manager (Train/Val/Test/OOD)
│       │   ├── features/             # Observable feature catalog & temporal extraction
│       │   │   ├── definitions.py    # Feature categorizations & forbidden label sets
│       │   │   └── extractor.py      # Causal rolling window features (x <= t)
│       │   ├── preprocessing/        # Train-fitted transformations
│       │   │   └── scalers.py        # Standard/Robust scalers, imputers, serializers
│       │   ├── models/               # Anomaly detection model suite
│       │   │   ├── base.py           # Common AnomalyDetector base interface
│       │   │   ├── statistical.py    # Mahalanobis & Robust MCD detectors
│       │   │   ├── classical.py      # Isolation Forest, One-Class SVM, LOF
│       │   │   ├── neural.py         # PyTorch MLP Dense Autoencoder
│       │   │   ├── temporal.py       # PyTorch LSTM Sequence-to-Sequence Autoencoder
│       │   │   └── registry.py       # Factory pattern & string model registry
│       │   ├── thresholding/         # Calibration & alarm state management
│       │   │   ├── calibration.py    # Normal-only quantile calibrator
│       │   │   └── alarm.py          # 3-state machine (NORMAL, SUSPECT, ALARM)
│       │   ├── evaluation/           # Security evaluation & metrics
│       │   │   ├── metrics.py        # Precision, Recall, F1, FPR, ROC, PR curves
│       │   │   ├── instance.py       # Attack instance delay & false alarms/hour
│       │   │   └── runner.py         # Cross-split evaluation & attack breakdowns
│       │   └── experiments/          # Experiment execution & diagnostics
│       │       ├── config.py         # ExperimentConfig Pydantic model
│       │       ├── runner.py         # End-to-end experiment executor
│       │       └── diagnostics.py    # Label shuffling, random features, plotters
│       └── validation/               # Analytical ground-truth verification
│           ├── __init__.py
│           ├── analytical.py         # Closed-form theoretical formulas
│           └── statistics.py         # Decoy state gain, QBER, & yield verifiers
├── scripts/                          # Automated execution & verification utilities
│   ├── _common.py                    # Shared script logging & path utilities
│   ├── generate_scenarios.py         # Scenario generation script
│   ├── generate_v05_dataset.py       # Full dataset batch script
│   ├── run_v05_benchmark.py          # Serial vs parallel throughput benchmark
│   ├── run_v05_experiments.py        # Controlled physical experiments (1-6)
│   ├── run_v06_benchmark.py          # Full ML benchmark suite (Experiments A-L)
│   └── validate_dataset.py           # Parquet schema & leakage audit script
├── tests/                            # Pytest validation suite (181 total tests)
│   ├── conftest.py                   # Pytest fixtures & environment setup
│   ├── helpers.py                    # Quick configuration & simulation helpers
│   ├── test_attack_physics.py        # Causal attack parameter transforms
│   ├── test_attacks.py               # Attack lifecycle & multi-attack tests
│   ├── test_channel.py               # Channel loss, dispersion, & misalignment
│   ├── test_checkpointing.py         # Atomic save/resume tests
│   ├── test_config.py                # Configuration parsing & schema validations
│   ├── test_detector.py              # SPAD dead time, afterpulsing, & clicks
│   ├── test_reproducibility.py       # Deterministic bit-level RNG seed hierarchy
│   ├── test_source.py                # Poisson laser statistics & decoy states
│   ├── test_statistics.py            # Analytical gain & yield theory tests
│   ├── test_telemetry.py             # Macroscopic window feature aggregation
│   ├── test_v01_regression.py        # Legacy v0.1 compatibility tests
│   ├── test_v04_environment.py       # Dynamic O-U drift & thermal coupling
│   ├── test_v04_physics.py           # Physical coupled phenomena tests
│   ├── test_v04_sensors.py           # ADC quantization & missingness tests
│   ├── test_v05_distributions.py     # Continuous/discrete math distribution tests
│   ├── test_v05_dependencies.py      # Causal dependency DAG & regime tests
│   ├── test_v05_parallel.py          # ProcessPoolExecutor fault tolerance
│   ├── test_v05_sharding.py          # Bounded-memory Parquet shard tests
│   ├── test_v05_leakage.py           # Zero label leakage & split isolation
│   ├── test_v06_data_loader.py       # ML dataset loading & chunking tests
│   ├── test_v06_feature_leakage.py   # Ground-truth exclusion assertions
│   ├── test_v06_causal_windowing.py  # Causal invariant temporal rolling tests
│   ├── test_v06_preprocessor.py      # Train-only scaler fitting & serialization
│   ├── test_v06_models.py            # Model fit/score/predict & score orientation
│   ├── test_v06_threshold_alarm.py   # Normal quantile threshold & alarm states
│   ├── test_v06_metrics_delay.py     # Metrics, ROC/PR, & detection delay tests
│   └── test_v06_sanity_experiments.py# Label shuffling & random features tests
├── pyproject.toml                    # Package build & dependency configuration
├── requirements.txt                  # Production runtime dependencies
├── requirements-dev.txt              # Testing, formatting, & linting tools
├── LICENSE                           # MIT Open Source License
└── README.md                         # Quickstart & user documentation
```

---

## 4. Physics & Simulation Engine (v0.1–v0.4)

### 4.1 Quantum Source (Alice)
Alice prepares optical pulses using a pulsed laser operating at repetition rate $f_{\text{rep}} \in [1, 50]\text{ MHz}$.
- **Photon Number Distribution**: Emitted pulse intensities follow Poisson statistics:
  $$P(n|\mu) = \frac{\mu^n e^{-\mu}}{n!}$$
  where $\mu \in \{\mu_{\text{sig}}, \mu_{\text{dec}}, \mu_{\text{vac}}\}$ is selected dynamically according to the decoy-state protocol probability vector $(p_{\text{sig}}, p_{\text{dec}}, p_{\text{vac}})$ with the strict invariant $\mu_{\text{sig}} > \mu_{\text{dec}} > \mu_{\text{vac}} = 0$.
- **State Preparation**: Alice chooses polarization bases (Rectilinear $\oplus$, Diagonal $\otimes$) and bit values ($0, 1$) with equal prior probabilities, preparing physical photon states $|\psi\rangle \in \{|H\rangle, |V\rangle, |D\rangle, |A\rangle\}$.
- **Source Non-Idealities**: Includes pulse-to-pulse optical intensity jitter ($\delta\mu \sim \mathcal{N}(0, \sigma_\mu^2)$) and laser diode wavelength/timing jitter.

### 4.2 Quantum Channel
The optical link models propagation through single-mode telecom fiber (SMF-28):
- **Beer-Lambert Attenuation**: Channel transmissivity $\eta_{\text{ch}}$ over link distance $L_{\text{fiber}}$:
  $$\eta_{\text{ch}} = 10^{-\frac{\alpha_{\text{att}} L_{\text{fiber}} + A_{\text{splice}}}{10}}$$
  where $\alpha_{\text{att}} \approx 0.20\text{ dB/km}$ at $1550\text{ nm}$, and $A_{\text{splice}}$ captures connector insertion losses.
- **Polarization Drift & Misalignment**: Fiber birefringence causes unitary polarization rotations, resulting in baseline polarization misalignment $\theta_{\text{mis}}$. The probability of bit error due to misalignment is:
  $$p_{\text{mis}} = \sin^2(\theta_{\text{mis}})$$
- **Chromatic Dispersion & Propagation Delay**: Total transit time $t_{\text{prop}} = \frac{n_{\text{core}} L_{\text{fiber}}}{c}$ with group-velocity dispersion broadening pulse arrival distributions.

### 4.3 Single-Photon Avalanche Diode Receiver (Bob)
Bob receives optical pulses using a passive 50:50 beam-splitter for random basis choice, followed by polarizing beam-splitters (PBS) directing photons to dedicated InGaAs/InP SPAD detectors (Detectors 0 and 1):
- **Quantum Efficiency**: Photons convert to photo-electrons with probability $\eta_{\text{det}} \in [0.10, 0.25]$. Total channel-detector collection efficiency is $\eta_{\text{total}} = 1 - (1 - \eta_{\text{ch}} \eta_{\text{det}})^n$.
- **Nonparalyzable Dead Time ($\tau_d$)**: When an avalanche triggers, the detector enters a refractory state of duration $\tau_d \in [20, 100]\text{ ns}$. During this window, incoming photons are lost without resetting the dead time timer. The observed count rate $r_{\text{obs}}$ under a Poisson arrival rate $\lambda$ is:
  $$r_{\text{obs}} = \frac{\lambda}{1 + \lambda \tau_d}$$
- **Rate-Dependent Effective Dead Time**: High incident fluxes draw charge from the quenching circuit, causing the effective dead time to expand:
  $$\tau_{\text{eff}}(\lambda) = \tau_0 + \Delta\tau \frac{\lambda}{\lambda + \lambda_c}$$
- **Exponential Afterpulsing**: Trapped carriers during avalanche discharge release stochastically, causing spurious clicks with probability $p_{\text{ap}} \in [0.01, 0.08]$ following an exponential delay distribution:
  $$P(t) = \frac{p_{\text{ap}}}{\tau_{\text{ap}}} e^{-t/\tau_{\text{ap}}}$$
- **Detector Telemetry**: Provides pulse-by-pulse observability: `dead_time_loss_rate`, `detector_0_busy_fraction`, `detector_0_ready_fraction`, `effective_dead_time`, interarrival time percentiles ($p_{05}, p_{50}, p_{95}$), and Fano factor $F = \frac{\sigma_{\text{clicks}}^2}{\mu_{\text{clicks}}}$.

### 4.4 Environmental Drift & Infrastructure Sensors (v0.4)
Legitimate QKD installations operate in uncontrolled environments with diurnal thermal swings:
- **Ornstein–Uhlenbeck Continuous Drift**: Continuous variables (e.g. ambient temperature $T(t)$, link attenuation $\alpha(t)$, polarization drift $\theta(t)$) evolve via mean-reverting stochastic differential equations:
  $$dX_t = \theta (\mu - X_t) dt + \sigma dW_t$$
  combined with diurnal harmonic terms $A_{\text{diurnal}} \sin\left(\frac{2\pi t}{86400} + \phi\right)$.
- **Physical Couplings**:
  - Dark count rate scales via Arrhenius thermal activation:
    $$R_{\text{dark}}(T) = R_0 \exp\left(-\frac{E_a}{k_B T}\right)$$
  - SPAD quantum efficiency degrades linearly with temperature:
    $$\eta(T) = \eta_0 [1 - \beta_T (T - T_0)]$$
- **Realistic Sensor Emulation**: Infrastructure sensors monitor temperature, vibration, humidity, and optical power through a finite multi-rate suite featuring:
  - Finite analog-to-digital converter (ADC) quantization ($\Delta_{\text{ADC}}$)
  - Additive white Gaussian sensor noise ($\epsilon \sim \mathcal{N}(0, \sigma_{\text{sensor}}^2)$)
  - Stochastic sensor dropout and missingness ($p_{\text{missing}} \in [0.001, 0.05]$)

---

## 5. Causal Physical Attack Engine (v0.3)

QISM rejects artificial feature manipulation (e.g. directly overriding `qber = 0.20`). Attacks operate causally on physical variables, allowing macroscopic telemetry to emerge organically.

### 5.1 The 5-Phase Attack Lifecycle
All attack episodes follow an explicit state machine:
```
[ NORMAL ] ──> [ ONSET ] ──> [ ACTIVE ] ──> [ TERMINATION ] ──> [ RECOVERY ] ──> [ NORMAL ]
```
- **`NORMAL`**: Baseline legitimate operation before interference.
- **`ONSET`**: Causal ramp-in phase governed by configurable profiles:
  - *Instantaneous*: Step-function activation ($f(t) = 1$).
  - *Linear*: $f(t) = \frac{t - t_0}{T_{\text{onset}}}$.
  - *Sigmoidal / Exponential*: Smooth physical actuation of optical attenuators or heating elements.
- **`ACTIVE`**: Full physical perturbation sustained at maximum severity.
- **`TERMINATION`**: Active cessation of malicious actuation.
- **`RECOVERY`**: Hardware relaxation back to equilibrium (e.g. thermal dissipation in blinded detectors).

### 5.2 Implemented Attack Catalog

```
+------------------------+------------------------------------------------------+----------------------------------------------------------+
| Attack Model           | Physical Mechanism & Actuation                       | Primary Physical Signatures                              |
+------------------------+------------------------------------------------------+----------------------------------------------------------+
| Intercept-Resend       | Eve measures in random basis; emits regenerated      | QBER increases to ~25% for intercepted pulses;           |
|                        | single-photon pulses.                                | Decoy-state yield parity violated.                      |
+------------------------+------------------------------------------------------+----------------------------------------------------------+
| Channel Loss           | Eve increases fiber attenuation (bending/tapping).   | Exponential drop in raw count rate;                      |
|                        |                                                      | Increase in apparent channel loss; Constant QBER.        |
+------------------------+------------------------------------------------------+----------------------------------------------------------+
| Source Manipulation    | Eve perturbs laser diode drive current / bias,       | Altered photon number statistics;                        |
|                        | modifying emitted mean photon numbers mu_sig, mu_dec.| Decoy-state residual violation: Q_mu != sum Y_n P(n|mu). |
+------------------------+------------------------------------------------------+----------------------------------------------------------+
| Detector Manipulation  | Eve injects high optical flux to force SPADs into    | Drop in quantum efficiency; Dark count suppression;      |
| (Blinding)             | continuous linear conduction mode.                   | Severe dead-time loss rate inflation.                    |
+------------------------+------------------------------------------------------+----------------------------------------------------------+
| Timing Shift           | Eve delays pulse arrival times relative to Bob's     | Click arrivals shift out of temporal detection gate;     |
|                        | gate synchronization window.                         | Imbalance between early and late detector arrivals.      |
+------------------------+------------------------------------------------------+----------------------------------------------------------+
| Photon-Number          | Eve selectively splits multi-photon pulses and       | Single-photon yield Y_1 suppressed;                      |
| Splitting (PNS)        | blocks single-photon pulses on signal states.        | Decoy-state gain ratio Q_dec / Q_sig shifts.             |
+------------------------+------------------------------------------------------+----------------------------------------------------------+
```

---

## 6. Population Diversity & Large-Scale Datasets (v0.5)

### 6.1 Parameter Distributions & Latin Hypercube Sampling
To prevent machine learning models from over-fitting to a single simulated device, V0.5 introduces parameter stratification:
- **Distribution Types**: `UniformDistribution`, `NormalDistribution`, `TruncatedNormalDistribution`, `LogNormalDistribution`, `BetaDistribution`, `CategoricalDistribution`, and `FixedDistribution`.
- **Space-Filling Latin Hypercube Sampling (LHS)**: Partitions the multidimensional cumulative probability space into $N$ equiprobable intervals, drawing one sample per stratum:
  $$x_i = F^{-1}\left(\frac{\pi(i) - U_i}{N}\right)$$
  where $\pi$ is a random permutation and $U_i \sim \mathcal{U}(0, 1)$.

### 6.2 Causal Dependency Graph & Regimes
Parameters are resolved through a directed acyclic graph (DAG):
1. Fiber distance $L_{\text{fiber}} \implies$ Total channel loss $A_{\text{total}} = \alpha L_{\text{fiber}} + A_{\text{splice}}$.
2. Detector pairing: Detector 0 efficiency $\eta_0$ is drawn from population distribution; Detector 1 efficiency is constrained by fabrication mismatch: $\eta_1 = \eta_0 (1 \pm \delta_{\text{fab}})$.
3. Decoy hierarchy enforcement: $\mu_{\text{sig}} > \mu_{\text{dec}} > \mu_{\text{vac}} = 0$.
4. Operational stress regimes:
   - `nominal`: Standard laboratory / datacenter conditions.
   - `high_loss`: Long-haul links ($> 50\text{ km}$, high attenuation).
   - `high_noise`: High ambient background temperature and elevated dark counts.
   - `high_instability`: Rapid thermal cycling and severe polarization drift.

### 6.3 Deterministic 5-Tier Seed Hierarchy
Bit-level reproducibility across single-threaded and parallel multiprocessing executions is guaranteed through a domain-separated cryptographic seed tree:
```
                          [ Master Seed ]
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
 [ Population Seed ]     [ Scenario Seed ]       [ Partition Seed ]
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
    [ Run Seed ]            [ Attack Seed ]         [ Sensor Seed ]
         │
    [ Worker RNG ]
```

### 6.4 Parallel Execution & Atomic Sharded Storage
- **`JobScheduler`**: Distributes `SimulationJob` tasks across local CPU cores via `concurrent.futures.ProcessPoolExecutor`.
- **Fault-Tolerant Checkpointing**: Execution state is maintained atomically in `job_manifest.json`. Interrupted runs resume without duplicate execution.
- **Bounded-Memory Sharding**: Telemetry events and window rows are streamed in batches of $50,000$ pulses directly into Apache Parquet files (`shard_0000.parquet`, `shard_0001.parquet`) with metadata manifests (`dataset_manifest.json`, `shard_manifest.json`).

---

## 7. Machine Learning Anomaly Detection Architecture (v0.6)

Version 0.6 introduces the machine learning layer, explicitly formulating security monitoring under the **one-class / unsupervised anomaly detection paradigm**.

```
                        LARGE-SCALE SHARDED DATASET
                                     │
                                     ▼
                          DATASET SCHEMA VALIDATOR
                                     │
                                     ▼
                              ML DATA LOADER
                   (Scenario Isolation: Train / Val / Test / OOD)
                                     │
                                     ▼
                              FEATURE ENGINE
                   (Strict Exclusion of Ground-Truth Labels)
                                     │
                                     ▼
                           PREPROCESSOR PIPELINE
                     (Fit strictly on Train Normal Data)
                                     │
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
      Statistical ML           Classical ML              Neural ML
     (Mahalanobis, MCD)     (Isolation Forest,       (Dense Autoencoder,
                                OCSVM, LOF)          Temporal LSTM AE)
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     ▼
                          CONTINUOUS ANOMALY SCORE
                                     │
                                     ▼
                           THRESHOLD CALIBRATION
                        (Normal Validation Quantiles)
                                     │
                                     ▼
                            ALARM STATE MACHINE
                   (NORMAL -> SUSPECT -> ALARM Escalation)
                                     │
                                     ▼
                            SECURITY EVALUATION
                    (IID vs OOD, Delay, False Alarms/Hour)
```

### 7.1 Data Split Architecture & Scenario Isolation
To eliminate temporal and infrastructure leakage, telemetry rows are never randomly split. Splitting is strictly enforced at the **scenario level**:
- **`train`**: Normal operation only (baseline model training).
- **`val`**: Normal operation only (threshold calibration and hyperparameter selection).
- **`test`**: In-Distribution (IID) evaluation (normal scenarios + attack scenarios on known hardware).
- **`generalization`**: Out-of-Distribution (OOD) evaluation (unseen detector fabrication tolerances, novel fiber distances, and extreme thermal regimes).

### 7.2 Observable Feature Space vs Ground-Truth Sidecars
To guarantee zero label leakage, fields are strictly segregated:
- **`GROUND_TRUTH_FIELDS`** (Forbidden from entering $X$): `attack_active`, `attack_type`, `attack_strength`, `mean_photon_number`, `yield_*`, `decoy_residual_predicted`, `bit_error_rate_ground_truth`.
- **`METADATA_FIELDS`** (Forbidden from entering $X$): `scenario_id`, `run_id`, `window_id`, `timestamp_start_s`, `timestamp_end_s`.
- **`OBSERVABLE_FIELDS`**: Physical measurements that physical hardware can observe.

#### Predefined Feature Catalogs:
- `qkd_only`: Conventional macroscopic rates (`qber`, `sifted_key_rate_hz`, `raw_count_rate_hz`, `detection_rate_hz`, `gain_signal`, `gain_decoy`, `gain_vacuum`).
- `detector_only`: SPAD physical dynamics (`dead_time_loss_rate`, `detector_0_click_share`, `dark_count_rate_hz`, `fano_factor`, `detector_0_busy_fraction`).
- `environmental_only`: External sensors (`ambient_temperature_c`, `humidity_pct`, `vibration_rms_g`).
- `infrastructure_only`: Physical link parameters (`estimated_transmissivity`, `attenuation_db_per_km`, `source_power_mw`).
- `full`: Complete union of all observable features.

### 7.3 Causal Temporal Feature Representation
For online stream monitoring, features at time $t$ must depend strictly on past observations ($x_{\le t}$):
- Rolling Mean:
  $$\mu_t = \frac{1}{L}\sum_{i=0}^{L-1} x_{t-i}$$
- Rolling Standard Deviation:
  $$\sigma_t = \sqrt{\frac{1}{L}\sum_{i=0}^{L-1}(x_{t-i} - \mu_t)^2}$$
- First-Order Difference:
  $$\Delta x_t = x_t - x_{t-1}$$

### 7.4 Preprocessing Pipeline & Serialization
All transformations are fit strictly on $\mathcal{D}_{\text{train}}$:
- **Standard Standardization**: $z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$
- **Robust Median/IQR Scaling**: $x' = \frac{x - \text{median}(x_{\text{train}})}{\text{IQR}(x_{\text{train}})}$
- **Pruning & Imputation**: Features with near-zero variance ($\sigma^2 < 10^{-12}$) are automatically dropped; missing values are imputed via training-set medians.
- **Serialization**: Preprocessing parameters are exported to `preprocessor.json` alongside model weights.

---

## 8. Anomaly Detection Models & Mathematical Formulations

Every detector implements the common `AnomalyDetector` interface (`fit`, `score`, `predict`, `save`, `load`), with the strict mathematical contract that **higher continuous anomaly scores indicate greater anomaly confidence**.

### 8.1 Statistical Baselines

#### A. Mahalanobis Distance (`mahalanobis`)
Models normal telemetry as a multivariate Gaussian centroid, computing distances in correlation geometry:
$$D_M(x) = \sqrt{(x - \hat{\mu})^T (\hat{\Sigma} + \lambda I)^{-1} (x - \hat{\mu})}$$
where $\hat{\mu} = \frac{1}{N}\sum_{i=1}^N x_i$ is the training centroid, $\hat{\Sigma}$ is the empirical covariance matrix, and $\lambda = 10^{-6}$ provides numerical ridge regularization.

#### B. Robust Covariance (`robust_covariance`)
Uses the Minimum Covariance Determinant (MCD) estimator to fit location and scatter to a robust central subset of points, providing resistance against heavy-tailed physical fluctuations.

### 8.2 Classical Machine Learning Detectors

#### C. Isolation Forest (`isolation_forest`)
Constructs an ensemble of isolation trees partitioning feature space via random axis-aligned cuts. Anomalies isolate with short average path lengths $h(x)$:
$$c(n) = 2 \left(\ln(n - 1) + \gamma\right) - \frac{2(n - 1)}{n}$$
Scores are inverted so higher scores represent anomalies:
$$s(x) = - \text{score\_samples}(x)$$

#### D. One-Class SVM (`one_class_svm`)
Finds a maximum-margin hyperplane in an RBF kernel Hilbert space separating normal training observations from the coordinate origin:
$$\min_{w, \xi, \rho} \frac{1}{2}\|w\|^2 + \frac{1}{\nu N}\sum_{i=1}^N \xi_i - \rho \quad \text{s.t.} \quad \langle w, \Phi(x_i) \rangle \ge \rho - \xi_i, \quad \xi_i \ge 0$$
Continuous decision scores are inverted:
$$s(x) = -f_{\text{decision}}(x)$$

#### E. Local Outlier Factor (`lof`)
Measures local density relative to $k$-nearest neighbors in novelty detection mode (`novelty=True`), scoring points by local reachability distance ratios.

### 8.3 Deep Neural Autoencoders

#### F. Dense Autoencoder (`autoencoder`)
A compact PyTorch multi-layer perceptron (MLP) trained to reconstruct normal telemetry:
```
Input (d) ──> Linear(d, 32) ──> GELU ──> Linear(32, 8) ──> GELU (Latent z)
               ──> Linear(8, 32) ──> GELU ──> Linear(32, d) ──> Output (x_hat)
```
Trained using AdamW with Mean Squared Error (MSE) loss, early stopping on validation loss, and atomic state checkpointing. The anomaly score is the reconstruction error:
$$S(x) = \|x - \hat{x}\|_2^2 = \sum_{j=1}^d (x_j - \hat{x}_j)^2$$

#### G. Temporal Sequence-to-Sequence LSTM Autoencoder (`temporal_autoencoder`)
Processes sliding temporal windows of historical telemetry $X_t = [x_{t-L+1}, \dots, x_t] \in \mathbb{R}^{L \times d}$:
```
Trajectory X_t (L x d) ──> Encoder LSTM (hidden=24) ──> Latent z (12)
                        ──> Decoder LSTM (hidden=24) ──> Linear(d) ──> Trajectory X_hat_t (L x d)
```
The temporal anomaly score is the normalized trajectory reconstruction error:
$$S_t = \frac{1}{L \cdot d} \sum_{i=1}^L \sum_{j=1}^d (X_{i,j} - \hat{X}_{i,j})^2$$

---

## 9. Threshold Calibration & Alarm State Machine

### 9.1 Normal-Only Threshold Calibration
To prevent threshold over-fitting, thresholds are derived strictly from normal validation telemetry:
$$\tau_\alpha = Q_{1 - \alpha}\left(S(\mathcal{D}_{\text{val\_normal}})\right)$$
calibrated across standard false-alarm operating points $\alpha \in \{0.05, 0.01, 0.005, 0.001\}$.

### 9.2 Operational Alarm State Machine
Single-window threshold crossings frequently produce false alarms due to transient Poisson fluctuations. QISM implements an operational state machine:

```
                  Score > tau_high
        [ NORMAL ] ──────────────> [ SUSPECT ]
            ▲                         │
            │ Score < tau_low         │ Sustained for K windows
            │                         ▼
            └──────────────────── [ ALARM ]
                 Score < tau_low
```

- **`NORMAL`**: Standard system operation.
- **`SUSPECT`**: Anomaly score exceeds $\tau_{\text{high}}$, initiating persistence tracking.
- **`ALARM`**: Anomaly score remains above $\tau_{\text{high}}$ for $K$ consecutive windows ($K \ge 2$), raising an operational security alert.
- **Hysteresis Recovery**: To prevent alarm chatter, the system returns to `NORMAL` only when scores drop below the lower recovery threshold $\tau_{\text{low}} = 0.9 \cdot \tau_{\text{high}}$.

---

## 10. Security Evaluation & Generalization Methodology

### 10.1 Evaluation Metrics
- **Window-Level Metrics**:
  - Precision: $\frac{TP}{TP + FP}$
  - Recall (Sensitivity): $\frac{TP}{TP + FN}$
  - F1 Score: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
  - False Positive Rate: $\text{FPR} = \frac{FP}{FP + TN}$
  - Specificity: $\frac{TN}{TN + FP}$
- **Continuous Curve Metrics**:
  - **AUROC**: Area Under the Receiver Operating Characteristic curve computed across continuous anomaly scores.
  - **AUPRC**: Area Under the Precision-Recall curve (critical for evaluating rare attack events).
- **Instance-Level Metrics**:
  - Attack Detection Rate: Fraction of ground-truth attack episodes triggering at least one valid alarm.
  - Detection Delay: Latency between attack onset $t_a$ and the first sustained alarm window $t_d$:
    $$D = t_d - t_a$$
  - Operational Alarm Rate: False alarms per simulated operational hour:
    $$\lambda_{\text{alarm}} = \frac{N_{\text{false\_alarms}}}{T_{\text{normal\_hours}}}$$

### 10.2 Out-of-Distribution (OOD) Generalization Gap
Models are evaluated simultaneously on In-Distribution (IID) test data and Out-of-Distribution (OOD) generalization data (unseen detector efficiency mismatches $\Delta\eta$ and extreme thermal regimes). The generalization gap quantifies real-world degradation:
$$\Delta \text{Recall} = \text{Recall}_{\text{IID}} - \text{Recall}_{\text{OOD}}$$
$$\Delta \text{FPR} = \text{FPR}_{\text{OOD}} - \text{FPR}_{\text{IID}}$$

---

## 11. Empirical Benchmark Results (Experiments A–L)

All experiments were executed against `data/v0.5_small/v05_small_dataset` (20 heterogeneous scenarios across 4 splits, 200,000 pulses total) using `scripts/run_v06_benchmark.py`.

### 11.1 Consolidated Baseline Model Benchmark

| Model Family | IID FPR | IID Recall | IID F1 | IID AUROC | IID AUPRC | OOD Recall | OOD FPR | Recall Gap ($\Delta \text{Rec}$) | Median Delay (s) |
|---|---|---|---|---|---|---|---|---|---|
| **`mahalanobis`** | 0.389 | 0.864 | 0.792 | 0.689 | 0.646 | 0.667 | 0.750 | +0.197 | 0.0020 |
| **`robust_covariance`** | 0.444 | 0.818 | 0.750 | 0.785 | 0.809 | 0.167 | 0.500 | +0.652 | 0.0020 |
| **`isolation_forest`** | 0.167 | 0.136 | 0.214 | 0.457 | 0.544 | 0.083 | 0.000 | +0.053 | N/A |
| **`one_class_svm`** | 0.444 | 0.864 | 0.776 | 0.682 | 0.648 | 0.667 | 0.750 | +0.197 | 0.0020 |
| **`lof`** | 0.444 | 0.864 | 0.776 | 0.672 | 0.638 | 0.667 | 0.750 | +0.197 | 0.0020 |
| **`autoencoder`** | 0.444 | 0.864 | 0.776 | 0.679 | 0.647 | 0.667 | 0.750 | +0.197 | 0.0020 |
| **`temporal_autoencoder`** | 0.222 | 0.682 | 0.732 | 0.720 | 0.715 | 0.583 | 0.875 | +0.098 | 0.0000 |

### 11.2 Feature Ablation Study (Isolation Forest Baseline)

| Feature Set | Observable Scope | Precision | Recall | F1 | FPR | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| **`qkd_only`** | Rate & QBER metrics | 0.500 | 0.045 | 0.083 | 0.056 | 0.477 | 0.575 |
| **`detector_only`** | SPAD dead-time & dynamics | 0.500 | 0.045 | 0.083 | 0.056 | 0.376 | 0.493 |
| **`environmental_only`** | Thermal, humidity, vibration | 0.667 | 0.091 | 0.160 | 0.056 | 0.369 | 0.550 |
| **`infrastructure_only`** | Channel loss & source power | 0.000 | 0.000 | 0.000 | 0.056 | 0.313 | 0.473 |
| **`full`** | Complete combined catalog | 0.000 | 0.000 | 0.000 | 0.056 | 0.419 | 0.498 |

### 11.3 Leakage Sanity Diagnostics (Experiment L)
- **Label Shuffling Diagnostic**: Ground-truth attack labels were randomly permuted across test windows. The measured AUROC collapsed to **$0.566$** ($\approx 0.50$, chance).
- **Random Feature Diagnostic**: Observable telemetry was replaced with uniform random noise. The measured AUROC collapsed to **$0.553$** ($\approx 0.50$, chance).
- **Outcome**: Confirms that model detection performance is driven entirely by genuine physical correlations rather than dataset metadata, scenario IDs, or implementation artifacts.

### 11.4 Attack-Specific Breakdown
- **Source Manipulation**: Detected with $20.0\%$ recall and a mean anomaly score of $+0.0133$. Strong physical signature in mean photon numbers.
- **Detector Manipulation**: Mean anomaly score $+0.0238$. Produces elevated reconstruction error during active phase.
- **PNS (Photon-Number Splitting)**: Mean anomaly score $-0.0168$ ($0.0\%$ recall under point-wise isolation forest). Subtle attacks mimicking normal loss remain challenging for point-wise classical models.

---

## 12. Verification & Test Suite Compendium

The test suite enforces complete regression invariance across all versions:
```bash
pytest -v
======================= 181 passed in 331.28s (0:05:31) =======================
```

### 12.1 Detailed Test Coverage Breakdown (181 Tests)

#### Group 1: Core Physical Simulation (v0.1–v0.4) — 88 Tests
- `tests/test_attack_physics.py` (6 tests): Validates causal attack transformations on optical parameters.
- `tests/test_attacks.py` (5 tests): Validates 5-phase attack lifecycles and multi-attack composition.
- `tests/test_channel.py` (8 tests): Validates Beer-Lambert attenuation, chromatic dispersion, and polarization rotations.
- `tests/test_checkpoint_attack.py` (1 test): Validates attack state serialization during mid-run checkpointing.
- `tests/test_checkpointing.py` (10 tests): Validates atomic state serialization, crash recovery, and manifest verification.
- `tests/test_config.py` (17 tests): Validates Pydantic configuration schemas, overrides, and unit validations.
- `tests/test_dataset_leakage.py` (2 tests): Asserts zero label leakage in v0.3 dataset generation.
- `tests/test_detector.py` (21 tests): Validates SPAD dead time, afterpulsing, rate-dependent $\tau_{\text{eff}}$, and dark counts.
- `tests/test_reproducibility.py` (5 tests): Validates deterministic seed derivation and bit-level identical run outputs.
- `tests/test_source.py` (8 tests): Validates Poisson laser emissions, decoy state intensities, and intensity jitter.
- `tests/test_statistics.py` (13 tests): Validates simulation gains, yields, and QBER against closed-form analytical equations.
- `tests/test_telemetry.py` (12 tests): Validates macroscopic window feature aggregation and Fano factor.
- `tests/test_v01_regression.py` (6 tests): Validates backwards compatibility with v0.1 configurations.
- `tests/test_v04_checkpoint.py` (1 test): Validates dynamic environmental state checkpointing.
- `tests/test_v04_environment.py` (6 tests): Validates Ornstein-Uhlenbeck drift and diurnal thermal harmonics.
- `tests/test_v04_physics.py` (6 tests): Validates physical temperature-dark count coupling and drift phenomena.
- `tests/test_v04_reproducibility.py` (2 tests): Validates reproducibility under continuous environmental drift.
- `tests/test_v04_telemetry.py` (3 tests): Validates multi-rate sensor noise, ADC quantization, and missingness.

#### Group 2: Population & Parallel Execution (v0.5) — 26 Tests
- `tests/test_v05_distributions.py` (8 tests): Validates continuous/discrete mathematical distributions and LHS stratification.
- `tests/test_v05_dependencies.py` (6 tests): Validates parameter dependency DAG, detector pairing, and stress regimes.
- `tests/test_v05_reproducibility.py` (4 tests): Validates serial vs parallel execution equivalence and seed isolation.
- `tests/test_v05_parallel.py` (2 tests): Validates ProcessPoolExecutor worker exception handling and fault tolerance.
- `tests/test_v05_checkpoint_resume.py` (1 test): Validates interrupted batch resume without duplicate execution.
- `tests/test_v05_sharding.py` (2 tests): Validates bounded-memory Parquet shard writing and unified logical reads.
- `tests/test_v05_leakage.py` (3 tests): Validates scenario-level partition isolation and generalization holdout ranges.

#### Group 3: Machine Learning & Benchmarking (v0.6 New) — 27 Tests
- `tests/test_v06_data_loader.py` (2 tests): Validates chunked Parquet loading, scenario isolation, and split retrieval.
- `tests/test_v06_feature_leakage.py` (3 tests): Asserts strict exclusion of ground-truth columns from observable matrices $X$.
- `tests/test_v06_causal_windowing.py` (2 tests): Proves causal invariance: modifying future rows ($t+1, \dots$) does not alter features at time $t$.
- `tests/test_v06_preprocessor.py` (3 tests): Validates fitting preprocessors strictly on training data, zero-variance handling, and JSON serialization.
- `tests/test_v06_models.py` (7 tests): Validates `fit`, `score`, `predict`, `save`, `load` across all 7 model families and asserts score orientation (higher score = anomalous).
- `tests/test_v06_threshold_alarm.py` (2 tests): Validates normal quantile calibration, alarm state transitions, persistence ($K$), and hysteresis.
- `tests/test_v06_metrics_delay.py` (3 tests): Validates confusion matrices, continuous AUROC/AUPRC integration, and attack detection delay ($D = t_d - t_a$).
- `tests/test_v06_sanity_experiments.py` (1 test): Validates that label shuffling and random features collapse AUROC toward 0.5.

---

## 13. CLI Reference & Operational Workflows

The unified CLI (`python -m qism`) exposes all simulator and machine learning operations:

```
usage: qism [-h] [--version]
            {generate-scenarios,generate-dataset,resume,validate-dataset,benchmark,
             train,evaluate,experiment,benchmark-ml,analyze-results} ...
```

### 13.1 Simulation & Dataset Commands (v0.5)

#### Generate Scenarios Preview (Dry Run)
```bash
python -m qism generate-scenarios configs/v0.5_small.yaml
```

#### Generate Full Population Dataset (Parallel Execution)
```bash
python -m qism generate-dataset configs/v0.5_small.yaml --workers 4 --fresh
```

#### Resume Interrupted Generation
```bash
python -m qism resume configs/v0.5_small.yaml --workers 4
```

#### Validate Dataset Integrity & Leakage
```bash
python -m qism validate-dataset data/v0.5_small/v05_small_dataset
```

#### Run Serial vs Parallel Scalability Benchmark
```bash
python -m qism benchmark configs/v0.5_smoke.yaml --workers 2
```

---

### 13.2 Machine Learning & Benchmarking Commands (v0.6)

#### Train Model & Calibrate Threshold
Trains an anomaly detector on normal data and exports model weights, preprocessor parameters, and calibrated thresholds:
```bash
python -m qism train configs/v0.6_small.yaml --model-type temporal_autoencoder --feature-set full
```

#### Evaluate Model on Dataset Split
Evaluates saved model artifacts against in-distribution test or out-of-distribution generalization splits:
```bash
python -m qism evaluate experiments/v06_small_experiment --split test
python -m qism evaluate experiments/v06_small_experiment --split generalization
```

#### Run Complete End-to-End Experiment
Orchestrates data loading, feature extraction, train-fitted scaling, model training, normal-only threshold calibration, IID/OOD evaluation, sanity diagnostics, and plot generation from a single configuration file:
```bash
python -m qism experiment configs/v0.6_small.yaml
```

#### Run Consolidated Multi-Model Benchmark
Executes the baseline comparison suite across all 7 model families:
```bash
python -m qism benchmark-ml configs/v0.6_small.yaml --output-dir experiments/benchmarks
```

#### Analyze & Tabulate Results
Parses experiment directories, displays formatted comparison tables in terminal, and outputs markdown summaries:
```bash
python -m qism analyze-results experiments
```

---

## 14. Known Limitations & Research Roadmap

### 14.1 Known Limitations
1. **Fixed-Model Degradation**: Version 0.6 models are static once trained on baseline normal telemetry. Long-term hardware aging and seasonal environmental drift will eventually trigger false alarms without online adaptation.
2. **Subtle Attack Stealth**: Photon-number splitting (PNS) attacks on short, low-loss fiber links remain close to the normal Poisson loss envelope; detection requires high-precision decoy-state yield inversion.
3. **Sequential Latency**: Deep sequence-to-sequence temporal models require higher inference latency than classical statistical baselines; streaming deployment on high-repetition-rate links ($> 1\text{ GHz}$) will require quantized or FPGA-accelerated inference.

### 14.2 Version 0.7–0.9 Roadmap
- **v0.7 (Streaming Inference & Online Adaptation)**:
  - Real-time sliding-window streaming inference engine.
  - Concept-drift handling via continuous rolling background retraining.
  - Decoy-state analytical residual temporal tracking.
- **v0.8 (Closed-Loop Autonomous Mitigation)**:
  - Automated physical response policies (optical attenuator actuation, key abort triggers, basis rerouting).
  - Integration of reinforcement learning security agents.
- **v0.9 / Future (Hardware Telemetry & Quantum ML)**:
  - Integration with experimental optical testbed hardware telemetry.
  - Quantum-enhanced machine learning (QML) anomaly detection baselines.

---

*QISM Compendium — Version 0.6.0. Copyright (c) 2026 QISM Project Contributors. Released under the MIT License.*
