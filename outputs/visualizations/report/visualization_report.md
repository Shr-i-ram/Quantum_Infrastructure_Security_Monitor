# QISM Simulator & Dataset Generation Methodology — Technical Presentation Report

**Baseline Simulator Version:** `0.5.0` (V0.1–V0.5 Architecture)
**Dataset Source:** `v05_small_dataset`
**Master RNG Seed:** `20260502`
**Generated:** `2026-09-08 19:05:45 UTC`

---

## 1. Simulator Overview & Architectural Philosophy

The Quantum Infrastructure Security Monitor (QISM) simulator models physical, stochastic, and temporal behavior of fiber-based decoy-state BB84 quantum key distribution systems. Unlike conventional analytical models or black-box statistical simulators, QISM enforces a **strict causal propagation invariant**:

$$\text{Physical Parameter Intervention} \longrightarrow \text{Quantum State & Channel Propagation} \longrightarrow \text{SPAD Detector Kinetics} \longrightarrow \text{Microscopic Events} \longrightarrow \text{Windowed Telemetry}$$

No telemetry feature or QBER value is ever altered synthetically. Every anomaly, metric deviation, or environmental drift emerges bottom-up from microscopic photon interactions, fiber attenuation, SPAD dead-time collisions, and afterpulse carrier dynamics.

---

## 2. Microscopic Event & Photon Generation Mechanics

Alice operates a weak coherent pulse (WCP) laser at repetition rate $f_{\mathrm{rep}} = 1\,\text{MHz}$ with randomized basis selection ($Z$ or $X$), bit preparation ($0$ or $1$), and three discrete intensity classes:
- **Signal State**: $\mu \approx 0.50$ (high single-photon probability)
- **Decoy State**: $\nu \approx 0.10$ (attenuated probe for photon-number-splitting detection)
- **Vacuum State**: $\omega = 0.0$ (dark count calibration)

Photon number per pulse follows the Poisson distribution:
$$P(N = n) = \frac{\mu^n e^{-\mu}}{n!}$$

---

## 3. Quantum Channel Loss Modeling & Dark Count Noise Floor

Photons propagate through a single-mode telecommunication fiber channel governed by Beer-Lambert exponential attenuation:
$$L_{\mathrm{total}} = \alpha L + L_{\mathrm{sys}} \quad [\mathrm{dB}], \qquad T(L) = 10^{-L_{\mathrm{total}} / 10}$$

where $\alpha \approx 0.18 - 0.22\,\text{dB/km}$ and $L_{\mathrm{sys}} \approx 1.0\,\text{dB}$ represents lumped splice and connector insertion losses. At long distances, signal detection decays exponentially until it collides with the constant thermal dark-count noise floor $2 R_{\mathrm{dark}}$, causing an inevitable rise in baseline QBER.

---

## 4. Bob SPAD Detector Dynamics & State Machine

Bob implements two Single-Photon Avalanche Diodes (D0 for bit 0, D1 for bit 1) operating in gated Geiger mode with temporal state dynamics:
1. **Nonparalyzable Dead Time**: Following an avalanche, the detector enters a dead interval of duration $\tau_{\mathrm{dead}}$ during which subsequent incident photons are discarded without extending dead time:
$$r_{\mathrm{obs}} = \frac{\lambda}{1 + \lambda \tau_{\mathrm{dead}}}$$
2. **Rate-Dependent Dead Time**: Heavy photon bombardment induces thermal and resistive heating, increasing effective dead time:
$$\tau_{\mathrm{eff}}(\lambda) = \tau_0 + \Delta \tau \frac{\lambda}{\lambda + \lambda_c}$$
3. **Exponential Afterpulsing**: Trapped charge carriers release delayed secondary avalanches:
$$P(\Delta t) = \frac{1}{\tau_{\mathrm{ap}}} \exp\left(-\frac{\Delta t}{\tau_{\mathrm{ap}}}\right)$$

---

## 5. Decoy-State BB84 Sifting & QBER Calculation

Classical post-processing sifts matching basis measurements ($Z_{\mathrm{Alice}} = Z_{\mathrm{Bob}}$ or $X_{\mathrm{Alice}} = X_{\mathrm{Bob}}$), discarding mismatched bases (~50% reduction). Quantum Bit Error Rate (QBER) is evaluated directly from sifted key bits:
$$\mathrm{QBER} = \frac{N_{\mathrm{error}}}{N_{\mathrm{sifted}}}$$

---

## 6. Physical Attack Injection & Telemetry Signatures

The simulator implements 6 distinct causal attack mechanisms with continuous transition profiles (Smoothstep Onset $\to$ Active $\to$ Recovery):
1. **Intercept-Resend**: Eve measures in random basis and resends; induces theoretical $\approx 25\%$ QBER increase in attacked pulses.
2. **Channel Loss**: Attenuation increase (+6 dB); suppresses signal and decoy gains while elevating relative dark-count share.
3. **Source Manipulation**: Pulse intensity perturbation $\Delta\mu$; distorts signal-to-decoy gain ratios.
4. **Detector Manipulation**: SPAD blinding / efficiency collapse; forces extreme count rate drops and detector imbalance.
5. **Timing Shift**: Arrival offset outside 1 ns gate; induces asymmetric count loss and timing jitter kurtosis.
6. **Photon Number Splitting (PNS)**: Selective multi-photon extraction; violates decoy state gain bounds.

---

## 7. Presentation Figures Index & Slide Provenance

| Figure ID | File Name | What It Shows | Key Analytical Formula | Slide Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Fig 01** | `01_simulation_pipeline.png` | End-to-end simulation flow from Alice through quantum channel, Bob SPAD receiver, sifting, and telemetry. | \text{Intervention} \to \text{Channel} \to \text{SPAD} \to \text{Events} \to \text{Telemetry} | Establishes causal bottom-up propagation invariant: physics drives events, which drive windowed telemetry. |
| **Fig 02** | `02_event_timeline.png` | Individual pulse-by-pulse state preparation (bits, bases, intensity), channel loss, detector clicks, and sifting. | N \sim \mathrm{Pois}(\mu) | Visualizes the microscopic realism of simulated individual optical events rather than macroscopic averages. |
| **Fig 03** | `03_channel_transmission_and_loss.png` | Beer-Lambert exponential attenuation law T(L) and expected detection rate decaying into thermal dark count floor. | T(L) = 10^{-(\alpha L + L_{\mathrm{sys}})/10}, \quad R_{\mathrm{det}}(L) = f_{\mathrm{rep}}\mu T(L)\eta_{\mathrm{det}} + 2R_{\mathrm{dark}} | Demonstrates why QBER intrinsically rises over long distances without an eavesdropper. |
| **Fig 04** | `04_detector_physics_and_dead_time.png` | Nonparalyzable dead time saturation, rate-dependent effective dead time model, afterpulse decay queue, and thermal dark drift. | r_{\mathrm{obs}} = \frac{\lambda}{1 + \lambda\tau_{\mathrm{dead}}}, \quad \tau_{\mathrm{eff}} = \tau_0 + \Delta\tau\frac{\lambda}{\lambda + \lambda_c} | Proves physical fidelity of Bob's receiver model: collisions reject photons nonparalyzably and generate secondary avalanches. |
| **Fig 05** | `05_basis_selection_and_sifting_funnel.png` | Basis correlation matrix (Z/X matching ~50%) and funnel reduction from 10k pulses to detected clicks, sifted key bits, and errors. | \mathrm{QBER} = \frac{N_{\mathrm{error}}}{N_{\mathrm{sifted}}} | Explains how raw physical photon pulses convert to cryptographic sifted key material and error metrics. |
| **Fig 06** | `06_normal_telemetry_distributions.png` | Empirical distributions of QBER, total counts, signal/decoy gains, detector imbalance, and sensor temperature. | X \sim \mathcal{D}_{\mathrm{normal}}(\mu, \sigma^2) | Demonstrates the statistical bounds of benign operation prior to any attack or anomaly. |
| **Fig 07** | `07_qber_temporal_evolution.png` | Continuous windowed QBER over time showing stable low error floor under benign drift versus sudden elevated deviation under attack. | \mathrm{QBER}(t) = \frac{N_{\mathrm{error}}(t)}{N_{\mathrm{sifted}}(t)} | Illustrates that QISM monitors temporal time series rather than a single static scalar number. |
| **Fig 08** | `08_attack_lifecycle_and_timeline.png` | Three-tiered aligned timeline showing ground truth attack phase, QBER excursion, and sifted key rate response. | g(u) = 3u^2 - 2u^3 | Proves that attacks are realistic physical processes with continuous ramp-up and recovery phases. |
| **Fig 09** | `09_multi_attack_signatures_comparison.png` | Multi-dimensional physical footprints of all 6 implemented attacks across QBER, counts, gains, timing, and imbalance. | \vec{X}_{\mathrm{attack}} - \vec{X}_{\mathrm{baseline}} | Demonstrates why multi-dimensional telemetry is vital: different attacks perturb entirely different subsets of observables. |
| **Fig 10** | `10_normal_vs_attack_feature_comparison.png` | Side-by-side boxplots of normal vs attack telemetry for QBER, counts, gains, detector imbalance, and Fano factor. | X_{\mathrm{normal}} \quad \text{vs} \quad X_{\mathrm{attack}} | Quantifies empirical separation between normal baseline and attack footprints across observable metrics. |
| **Fig 11** | `11_telemetry_cross_correlation.png` | Heatmap of Pearson correlations across 10 observable telemetry variables. | r = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x-\bar{x})^2 \sum (y-\bar{y})^2}} | Demonstrates realistic physical coupling between temperature, dark counts, dead time loss, and optical gains. |
| **Fig 12** | `12_dataset_partition_composition.png` | Bar and donut charts of scenario runs and pulses across Calibration, Normal, Attack, and Generalization partitions. | N_{\mathrm{scenarios}} = 20, \quad N_{\mathrm{pulses}} = 200{,}000 | Explains how the simulation data is organized into standardized splits for rigorous evaluation. |
| **Fig 13** | `13_population_parameter_distributions.png` | Distributions of fiber distance, detector efficiency, dark count rate, and ambient temperature across scenarios. | \theta_i \sim \mathrm{LHS}(\Omega_{\mathrm{param}}) | Shows that scenarios are systematically sampled across physical operational ranges via Latin Hypercube Sampling. |
| **Fig 14** | `14_generalization_distribution_shift.png` | 2D parameter space comparing in-distribution scenarios vs extreme held-out distance and degraded efficiency regimes. | \mathrm{dist}(x_{\mathrm{gen}}, \Omega_{\mathrm{train}}) > \delta | Demonstrates the simulator's capability to generate out-of-distribution environments for robustness evaluation. |
| **Fig 15** | `15_stochastic_consistency_and_reproducibility.png` | Exact bit-level match between identical seeds (MAE = 0.0) alongside independent seeds fluctuating consistently around physical mean. | \mathcal{S}(\text{seed}) = \text{Deterministic Output} | Proves scientific reproducibility and validates that Monte Carlo variance respects statistical consistency. |

---

## 8. Presentation Delivery Notes

When presenting these figures to technical audiences:
1. **Emphasize Causality**: Point out that anomalies are not artificially drawn; they propagate from physical parameters through photon events to telemetry.
2. **Highlight Multi-Dimensionality**: Show that while single-variable QBER is a scalar, the 130-feature telemetry vector exposes subtle attacks that QBER alone cannot reveal.
3. **Distinguish Baseline vs ML**: Remind listeners that this presentation covers the physical simulator and dataset generation methodology (V0.1–V0.5), providing the foundational ground truth before any ML anomaly detector is applied.