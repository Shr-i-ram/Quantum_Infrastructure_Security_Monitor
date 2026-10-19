"""QISM Visualization Subsystem (V0.1–V0.5 Baseline).

Generates publication-grade, presentation-quality visualizations illustrating
the physical QKD simulation pipeline, channel loss laws, SPAD detector kinetics,
causal attack footprints, normal telemetry distributions, and dataset construction.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

from qism.visualization.config import VisualizationConfig
from qism.visualization.loaders import (
    discover_runs,
    get_representative_run,
    load_all_telemetry,
    load_dataset_manifest,
    load_run_events,
    load_run_telemetry,
    load_scenario_manifest,
    load_split_telemetry,
)
from qism.visualization.style import apply_presentation_style

# Plotting modules
from qism.visualization.channel_plots import plot_channel_transmission
from qism.visualization.detector_plots import plot_detector_physics
from qism.visualization.event_plots import plot_event_timeline, plot_timing_jitter_and_arrival
from qism.visualization.qkd_plots import plot_basis_sifting_funnel, plot_simulation_pipeline
from qism.visualization.attack_plots import (
    plot_attack_lifecycle,
    plot_multi_attack_signatures,
    plot_normal_vs_attack_comparison,
)
from qism.visualization.telemetry_plots import (
    plot_normal_telemetry_distributions,
    plot_qber_temporal_evolution,
    plot_telemetry_correlation_matrix,
)
from qism.visualization.dataset_plots import (
    plot_dataset_composition,
    plot_generalization_shift,
    plot_population_distributions,
    plot_seed_reproducibility,
)
from qism.visualization.report import compile_figure_metadata, generate_visualization_report

log = logging.getLogger("qism.visualization")

__all__ = [
    "VisualizationConfig",
    "generate_all_visualizations",
    "apply_presentation_style",
    "plot_simulation_pipeline",
    "plot_event_timeline",
    "plot_channel_transmission",
    "plot_detector_physics",
    "plot_basis_sifting_funnel",
    "plot_normal_telemetry_distributions",
    "plot_qber_temporal_evolution",
    "plot_attack_lifecycle",
    "plot_multi_attack_signatures",
    "plot_normal_vs_attack_comparison",
    "plot_telemetry_correlation_matrix",
    "plot_dataset_composition",
    "plot_population_distributions",
    "plot_generalization_shift",
    "plot_seed_reproducibility",
]


def generate_all_visualizations(config: VisualizationConfig | None = None) -> dict[str, Any]:
    """Execute end-to-end visualization generation across all categories.

    Produces:
        - 15 Presentation Figures in outputs/visualizations/presentation/
        - Exploratory Figures in outputs/visualizations/exploratory/
        - Figure Metadata tracking in outputs/visualizations/metadata/
        - Markdown Presentation Report in outputs/visualizations/report/
    """
    if config is None:
        config = VisualizationConfig()

    apply_presentation_style()
    config.ensure_directories()

    ds_dir = config.dataset_path
    log.info("Loading dataset manifests and runs from %s...", ds_dir)

    dataset_manifest = load_dataset_manifest(ds_dir)
    scenario_manifest = load_scenario_manifest(ds_dir)
    all_runs = discover_runs(ds_dir)
    log.info("Discovered %d runs across dataset partitions.", len(all_runs))

    # Load partition telemetry
    normal_df = load_split_telemetry(ds_dir, "normal")
    attack_df = load_split_telemetry(ds_dir, "attack")
    all_tel_df = load_all_telemetry(ds_dir)

    # Load representative events
    sample_normal_run = get_representative_run(ds_dir, "normal")
    events_df = None
    if sample_normal_run is not None:
        try:
            events_df = load_run_events(sample_normal_run, max_pulses=config.max_event_samples)
            log.info("Loaded %d events from representative normal run %s.", len(events_df), sample_normal_run.name)
        except Exception as e:
            log.warning("Could not load events from %s: %s", sample_normal_run, e)

    sample_attack_run = get_representative_run(ds_dir, "attack")
    attack_run_tel = None
    if sample_attack_run is not None:
        try:
            attack_run_tel = load_run_telemetry(sample_attack_run)
            log.info("Loaded attack telemetry from %s.", sample_attack_run.name)
        except Exception as e:
            log.warning("Could not load telemetry from %s: %s", sample_attack_run, e)

    figures_info: list[dict[str, Any]] = []

    # =========================================================================
    # GENERATE PRESENTATION FIGURES (1 to 15)
    # =========================================================================
    p_dir = config.presentation_dir
    fmts = config.formats
    dpi = config.dpi

    log.info("Generating Curated Presentation Figures in %s...", p_dir)

    # Figure 01: Simulation Pipeline Architecture Flow
    res = plot_simulation_pipeline(p_dir, formats=fmts, dpi=dpi)
    figures_info.append({
        "id": "Fig 01",
        "filename": "01_simulation_pipeline.png",
        "title": "QISM Simulation Pipeline & Physical Architecture",
        "what_it_shows": "End-to-end simulation flow from Alice through quantum channel, Bob SPAD receiver, sifting, and telemetry.",
        "why_it_matters": "Establishes causal bottom-up propagation invariant: physics drives events, which drive windowed telemetry.",
        "formula": r"\text{Intervention} \to \text{Channel} \to \text{SPAD} \to \text{Events} \to \text{Telemetry}",
        "paths": {k: str(v) for k, v in res.items()},
    })

    # Figure 02: Event-Level Microscopic Raster Timeline
    if events_df is not None and not events_df.empty:
        res = plot_event_timeline(events_df, p_dir, formats=fmts, dpi=dpi)
        figures_info.append({
            "id": "Fig 02",
            "filename": "02_event_timeline.png",
            "title": "Event-Level Microscopic Pulse Timeline",
            "what_it_shows": "Individual pulse-by-pulse state preparation (bits, bases, intensity), channel loss, detector clicks, and sifting.",
            "why_it_matters": "Visualizes the microscopic realism of simulated individual optical events rather than macroscopic averages.",
            "formula": r"N \sim \mathrm{Pois}(\mu)",
            "paths": {k: str(v) for k, v in res.items()},
        })

    # Figure 03: Channel Attenuation Law & Distance Scaling
    res = plot_channel_transmission(scenario_manifest, all_tel_df, p_dir, formats=fmts, dpi=dpi)
    figures_info.append({
        "id": "Fig 03",
        "filename": "03_channel_transmission_and_loss.png",
        "title": "Quantum Channel Loss & Detection Rate Floor",
        "what_it_shows": "Beer-Lambert exponential attenuation law T(L) and expected detection rate decaying into thermal dark count floor.",
        "why_it_matters": "Demonstrates why QBER intrinsically rises over long distances without an eavesdropper.",
        "formula": r"T(L) = 10^{-(\alpha L + L_{\mathrm{sys}})/10}, \quad R_{\mathrm{det}}(L) = f_{\mathrm{rep}}\mu T(L)\eta_{\mathrm{det}} + 2R_{\mathrm{dark}}",
        "paths": {k: str(v) for k, v in res.items()},
    })

    # Figure 04: SPAD Detector Physics & Kinetics
    res = plot_detector_physics(p_dir, formats=fmts, dpi=dpi)
    figures_info.append({
        "id": "Fig 04",
        "filename": "04_detector_physics_and_dead_time.png",
        "title": "SPAD Detector Dynamics: Saturation, Dead Time & Afterpulsing",
        "what_it_shows": "Nonparalyzable dead time saturation, rate-dependent effective dead time model, afterpulse decay queue, and thermal dark drift.",
        "why_it_matters": "Proves physical fidelity of Bob's receiver model: collisions reject photons nonparalyzably and generate secondary avalanches.",
        "formula": r"r_{\mathrm{obs}} = \frac{\lambda}{1 + \lambda\tau_{\mathrm{dead}}}, \quad \tau_{\mathrm{eff}} = \tau_0 + \Delta\tau\frac{\lambda}{\lambda + \lambda_c}",
        "paths": {k: str(v) for k, v in res.items()},
    })

    # Figure 05: Basis Selection Matrix & Sifting Funnel
    if events_df is not None and not events_df.empty:
        res = plot_basis_sifting_funnel(events_df, p_dir, formats=fmts, dpi=dpi)
        figures_info.append({
            "id": "Fig 05",
            "filename": "05_basis_selection_and_sifting_funnel.png",
            "title": "BB84 Basis Matching Matrix & Sifting Funnel",
            "what_it_shows": "Basis correlation matrix (Z/X matching ~50%) and funnel reduction from 10k pulses to detected clicks, sifted key bits, and errors.",
            "why_it_matters": "Explains how raw physical photon pulses convert to cryptographic sifted key material and error metrics.",
            "formula": r"\mathrm{QBER} = \frac{N_{\mathrm{error}}}{N_{\mathrm{sifted}}}",
            "paths": {k: str(v) for k, v in res.items()},
        })

    # Figure 06: Normal Telemetry Baseline Distributions
    if not normal_df.empty:
        res = plot_normal_telemetry_distributions(normal_df, p_dir, formats=fmts, dpi=dpi)
        figures_info.append({
            "id": "Fig 06",
            "filename": "06_normal_telemetry_distributions.png",
            "title": "Normal Operation Multi-Dimensional Baseline Distributions",
            "what_it_shows": "Empirical distributions of QBER, total counts, signal/decoy gains, detector imbalance, and sensor temperature.",
            "why_it_matters": "Demonstrates the statistical bounds of benign operation prior to any attack or anomaly.",
            "formula": r"X \sim \mathcal{D}_{\mathrm{normal}}(\mu, \sigma^2)",
            "paths": {k: str(v) for k, v in res.items()},
        })

    # Figure 07: QBER Temporal Evolution (Normal vs Attack)
    res = plot_qber_temporal_evolution(normal_df, attack_df, p_dir, formats=fmts, dpi=dpi)
    figures_info.append({
        "id": "Fig 07",
        "filename": "07_qber_temporal_evolution.png",
        "title": "Temporal QBER Dynamics: Benign Regime vs Attack Anomaly",
        "what_it_shows": "Continuous windowed QBER over time showing stable low error floor under benign drift versus sudden elevated deviation under attack.",
        "why_it_matters": "Illustrates that QISM monitors temporal time series rather than a single static scalar number.",
        "formula": r"\mathrm{QBER}(t) = \frac{N_{\mathrm{error}}(t)}{N_{\mathrm{sifted}}(t)}",
        "paths": {k: str(v) for k, v in res.items()},
    })

    # Figure 08: Attack Lifecycle Timeline (Onset -> Active -> Recovery)
    lifecycle_df = attack_run_tel if attack_run_tel is not None and not attack_run_tel.empty else attack_df
    if not lifecycle_df.empty:
        res = plot_attack_lifecycle(lifecycle_df, p_dir, formats=fmts, dpi=dpi)
        figures_info.append({
            "id": "Fig 08",
            "filename": "08_attack_lifecycle_and_timeline.png",
            "title": "Causal Attack Lifecycle: Smoothstep Onset, Active & Recovery",
            "what_it_shows": "Three-tiered aligned timeline showing ground truth attack phase, QBER excursion, and sifted key rate response.",
            "why_it_matters": "Proves that attacks are realistic physical processes with continuous ramp-up and recovery phases.",
            "formula": r"g(u) = 3u^2 - 2u^3",
            "paths": {k: str(v) for k, v in res.items()},
        })

    # Figure 09: Multi-Attack Signature Profiles Across 6 Attacks
    res = plot_multi_attack_signatures(p_dir, formats=fmts, dpi=dpi)
    figures_info.append({
        "id": "Fig 09",
        "filename": "09_multi_attack_signatures_comparison.png",
        "title": "Multi-Attack Signatures Across Telemetry Dimensions",
        "what_it_shows": "Multi-dimensional physical footprints of all 6 implemented attacks across QBER, counts, gains, timing, and imbalance.",
        "why_it_matters": "Demonstrates why multi-dimensional telemetry is vital: different attacks perturb entirely different subsets of observables.",
        "formula": r"\vec{X}_{\mathrm{attack}} - \vec{X}_{\mathrm{baseline}}",
        "paths": {k: str(v) for k, v in res.items()},
    })

    # Figure 10: Normal vs Attack Feature Comparison
    if not normal_df.empty and not attack_df.empty:
        res = plot_normal_vs_attack_comparison(normal_df, attack_df, p_dir, formats=fmts, dpi=dpi)
        figures_info.append({
            "id": "Fig 10",
            "filename": "10_normal_vs_attack_feature_comparison.png",
            "title": "Feature Distribution Shift: Normal vs Attack Telemetry",
            "what_it_shows": "Side-by-side boxplots of normal vs attack telemetry for QBER, counts, gains, detector imbalance, and Fano factor.",
            "why_it_matters": "Quantifies empirical separation between normal baseline and attack footprints across observable metrics.",
            "formula": r"X_{\mathrm{normal}} \quad \text{vs} \quad X_{\mathrm{attack}}",
            "paths": {k: str(v) for k, v in res.items()},
        })

    # Figure 11: Telemetry Cross-Correlation Heatmap
    if not all_tel_df.empty:
        res = plot_telemetry_correlation_matrix(all_tel_df, p_dir, formats=fmts, dpi=dpi)
        figures_info.append({
            "id": "Fig 11",
            "filename": "11_telemetry_cross_correlation.png",
            "title": "Telemetry Cross-Correlation Matrix & Physical Covariance",
            "what_it_shows": "Heatmap of Pearson correlations across 10 observable telemetry variables.",
            "why_it_matters": "Demonstrates realistic physical coupling between temperature, dark counts, dead time loss, and optical gains.",
            "formula": r"r = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x-\bar{x})^2 \sum (y-\bar{y})^2}}",
            "paths": {k: str(v) for k, v in res.items()},
        })

    # Figure 12: Dataset Partition Composition
    res = plot_dataset_composition(dataset_manifest, all_runs, p_dir, formats=fmts, dpi=dpi)
    figures_info.append({
        "id": "Fig 12",
        "filename": "12_dataset_partition_composition.png",
        "title": "Dataset Architecture & Functional Partition Breakdown",
        "what_it_shows": "Bar and donut charts of scenario runs and pulses across Calibration, Normal, Attack, and Generalization partitions.",
        "why_it_matters": "Explains how the simulation data is organized into standardized splits for rigorous evaluation.",
        "formula": r"N_{\mathrm{scenarios}} = 20, \quad N_{\mathrm{pulses}} = 200{,}000",
        "paths": {k: str(v) for k, v in res.items()},
    })

    # Figure 13: Population Parameter Space Coverage (LHS)
    res = plot_population_distributions(scenario_manifest, p_dir, formats=fmts, dpi=dpi)
    figures_info.append({
        "id": "Fig 13",
        "filename": "13_population_parameter_distributions.png",
        "title": "Population Diversity: Latin Hypercube Parameter Coverage",
        "what_it_shows": "Distributions of fiber distance, detector efficiency, dark count rate, and ambient temperature across scenarios.",
        "why_it_matters": "Shows that scenarios are systematically sampled across physical operational ranges via Latin Hypercube Sampling.",
        "formula": r"\theta_i \sim \mathrm{LHS}(\Omega_{\mathrm{param}})",
        "paths": {k: str(v) for k, v in res.items()},
    })

    # Figure 14: Generalization Distribution Shift (Held-Out Regimes)
    res = plot_generalization_shift(scenario_manifest, p_dir, formats=fmts, dpi=dpi)
    figures_info.append({
        "id": "Fig 14",
        "filename": "14_generalization_distribution_shift.png",
        "title": "Generalization Regimes: Held-Out Infrastructure Parameters",
        "what_it_shows": "2D parameter space comparing in-distribution scenarios vs extreme held-out distance and degraded efficiency regimes.",
        "why_it_matters": "Demonstrates the simulator's capability to generate out-of-distribution environments for robustness evaluation.",
        "formula": r"\mathrm{dist}(x_{\mathrm{gen}}, \Omega_{\mathrm{train}}) > \delta",
        "paths": {k: str(v) for k, v in res.items()},
    })

    # Figure 15: Stochastic Consistency & Seed Reproducibility
    res = plot_seed_reproducibility(normal_df, p_dir, formats=fmts, dpi=dpi)
    figures_info.append({
        "id": "Fig 15",
        "filename": "15_stochastic_consistency_and_reproducibility.png",
        "title": "RNG Architecture: Bit-Exact Reproducibility & Stochastic Variance",
        "what_it_shows": "Exact bit-level match between identical seeds (MAE = 0.0) alongside independent seeds fluctuating consistently around physical mean.",
        "why_it_matters": "Proves scientific reproducibility and validates that Monte Carlo variance respects statistical consistency.",
        "formula": r"\mathcal{S}(\text{seed}) = \text{Deterministic Output}",
        "paths": {k: str(v) for k, v in res.items()},
    })

    # =========================================================================
    # GENERATE EXPLORATORY FIGURES
    # =========================================================================
    e_dir = config.exploratory_dir
    log.info("Generating Exploratory Diagnostic Figures in %s...", e_dir)
    if events_df is not None and not events_df.empty:
        plot_timing_jitter_and_arrival(events_df, e_dir, formats=fmts, dpi=dpi)

    # =========================================================================
    # COMPILE METADATA & REPORT
    # =========================================================================
    log.info("Compiling metadata and presentation report...")
    meta_path = compile_figure_metadata(figures_info, config.metadata_dir, dataset_manifest)
    report_path = generate_visualization_report(figures_info, config.report_dir, dataset_manifest)

    log.info("Successfully generated %d presentation figures, metadata, and report.", len(figures_info))

    return {
        "status": "COMPLETED",
        "total_presentation_figures": len(figures_info),
        "presentation_dir": str(config.presentation_dir),
        "exploratory_dir": str(config.exploratory_dir),
        "metadata_file": str(meta_path),
        "report_file": str(report_path),
        "figures": figures_info,
    }
