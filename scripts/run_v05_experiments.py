#!/usr/bin/env python3
"""Execute the 6 required controlled physical experiments for QISM V0.5 (Requirement 200).

Experiments:
1. Fixed Configuration: Multiple runs from one fixed scenario -> within-device stochastic variation.
2. Population Variation: Multiple scenarios from population -> between-device infrastructure variation.
3. Environmental Variation: Fixed infrastructure, changing environmental baseline -> natural telemetry drift.
4. Infrastructure Variation: Fixed environment, varying device/channel parameters -> physical scaling.
5. Attack Variation: Fixed infrastructure, varied attack types/severities/timings -> distinct signatures.
6. Combined Variation: Heterogeneous infrastructure + environment + attack conditions -> population diversity.

Usage:
    python scripts/run_v05_experiments.py
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import sys
from typing import Any
import numpy as np

# Ensure src is on sys.path
_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qism.config import (
    AttackScenarioConfig,
    ParameterDistributionConfig,
    PopulationConfig,
    QKDConfig,
    Scenario,
    SplitConfig,
    apply_overrides,
    default_config,
)
from qism.execution.job import SimulationJob
from qism.execution.worker import execute_simulation_job
from qism.population.generator import ScenarioGenerator
from qism.random import derive_run_seed, derive_scenario_seed

logging.basicConfig(level=logging.WARNING)


def exp1_fixed_configuration() -> dict[str, Any]:
    print("=" * 70)
    print("EXPERIMENT 1: Fixed Configuration -> Within-Device Stochastic Variation")
    print("=" * 70)

    base = default_config()
    pulses = 10000
    overrides = {
        "protocol.n_pulses": pulses,
        "telemetry.window_size": 2500,
        "simulation.chunk_size": 2500,
        "simulation.checkpoint_interval": 2500,
        "channel.distance_km": 20.0,
        "detectors.detector_0.efficiency": 0.15,
        "detectors.detector_1.efficiency": 0.15,
        "source.mu_signal": 0.50,
        "attack.enabled": False,
        "attack.scenarios": [],
    }
    cfg = apply_overrides(base, overrides)
    scen = Scenario(
        scenario_id="exp1_fixed_scen",
        infrastructure_id="dev_fixed_01",
        seed=12345,
        regime="nominal",
        config=cfg,
    )

    out_dir = pathlib.Path("data/v05_experiments/exp1")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    detection_counts = []
    qber_values = []

    for run_idx in range(4):
        run_id = f"run_{run_idx:03d}"
        run_seed = derive_run_seed(scen.seed, run_id)
        job = SimulationJob(
            scenario=scen,
            run_id=run_id,
            seed=run_seed,
            output_dir=str(out_dir),
            checkpoint_dir=str(out_dir / "checkpoints"),
        )
        res = execute_simulation_job(job)
        assert res.status == "COMPLETED", f"Job failed: {res.error_message}"
        results.append(res)
        det_cnt = res.metrics.get("raw_detections", 0)
        qber = res.metrics.get("sifted_qber", 0.0)
        detection_counts.append(det_cnt)
        qber_values.append(qber)
        print(f"  Run {run_idx}: seed={run_seed} detections={det_cnt} sifted_QBER={qber:.4f}")

    mean_det = float(np.mean(detection_counts))
    std_det = float(np.std(detection_counts))
    cv_det = std_det / mean_det if mean_det > 0 else 0.0

    print(f"  Detection count: mean={mean_det:.1f}, std={std_det:.1f}, CV={cv_det:.4f}")
    # Physical sanity: within-device CV is small but non-zero (stochastic Poisson variance)
    passed = cv_det > 0.0 and cv_det < 0.10
    print(f"  Experiment 1: {'PASSED' if passed else 'FAILED'}")
    return {
        "experiment": "1_fixed_configuration",
        "passed": passed,
        "detection_counts": detection_counts,
        "mean_detections": mean_det,
        "std_detections": std_det,
        "cv_detections": cv_det,
        "qber_values": qber_values,
    }


def exp2_population_variation() -> dict[str, Any]:
    print("=" * 70)
    print("EXPERIMENT 2: Population Variation -> Between-Device Infrastructure Diversity")
    print("=" * 70)

    pop_cfg = PopulationConfig(
        channel={
            "distance_km": ParameterDistributionConfig(
                distribution="uniform", a=10.0, b=40.0, min_val=5.0, max_val=50.0, unit="km"
            )
        },
        detectors={
            "efficiency": ParameterDistributionConfig(
                distribution="truncated_normal", mean=0.18, std=0.03, min_val=0.10, max_val=0.30
            )
        },
    )
    generator = ScenarioGenerator(pop_cfg, master_seed=999)
    split_cfg = SplitConfig(n_scenarios=4, runs_per_scenario=1, pulses_per_run=10000, edge_case_fraction=0.0, attack_fraction=0.0)
    scenarios = generator.generate_split_scenarios("pop_test", split_cfg, split_seed=5555)

    out_dir = pathlib.Path("data/v05_experiments/exp2")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    efficiencies = []
    distances = []
    detections = []

    for scen in scenarios:
        run_seed = derive_run_seed(scen.seed, "run_000")
        job = SimulationJob(
            scenario=scen,
            run_id="run_000",
            seed=run_seed,
            output_dir=str(out_dir),
            checkpoint_dir=str(out_dir / "checkpoints"),
        )
        res = execute_simulation_job(job)
        assert res.status == "COMPLETED"
        eff = scen.config.detectors.detector_0.efficiency
        dist = scen.config.channel.distance_km
        det = res.metrics.get("raw_detections", 0)
        efficiencies.append(eff)
        distances.append(dist)
        detections.append(det)
        print(f"  Scen {scen.scenario_id}: dist={dist:.1f}km eff={eff:.3f} -> detections={det}")

    # Population variation should produce wider variance across hardware than within-device
    det_range = max(detections) - min(detections)
    eff_range = max(efficiencies) - min(efficiencies)
    passed = eff_range > 0.02 and det_range > 50
    print(f"  Efficiency range={eff_range:.3f}, Detection range={det_range}")
    print(f"  Experiment 2: {'PASSED' if passed else 'FAILED'}")
    return {
        "experiment": "2_population_variation",
        "passed": passed,
        "efficiencies": efficiencies,
        "distances": distances,
        "detections": detections,
        "eff_range": eff_range,
        "det_range": det_range,
    }


def exp3_environmental_variation() -> dict[str, Any]:
    print("=" * 70)
    print("EXPERIMENT 3: Environmental Variation -> Natural Telemetry Drift")
    print("=" * 70)

    base = default_config()
    pulses = 10000
    temperatures = [18.0, 25.0, 32.0]
    out_dir = pathlib.Path("data/v05_experiments/exp3")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dark_counts = []

    for t_val in temperatures:
        overrides = {
            "protocol.n_pulses": pulses,
            "telemetry.window_size": 2500,
            "environment.type": "dynamic",
            "environment.enabled": True,
            "environment.temperature.baseline": t_val,
            "environment.temperature.theta": 0.5,
            "environment.temperature.sigma": 0.1,
            "detectors.detector_0.temp_coeff_dark_count": 0.03,
            "detectors.detector_1.temp_coeff_dark_count": 0.03,
            "attack.enabled": False,
        }
        cfg = apply_overrides(base, overrides)
        scen = Scenario(
            scenario_id=f"scen_temp_{int(t_val)}",
            infrastructure_id="dev_env_01",
            seed=4444,
            config=cfg,
        )
        job = SimulationJob(
            scenario=scen,
            run_id="run_000",
            seed=derive_run_seed(scen.seed, "run_000"),
            output_dir=str(out_dir),
            checkpoint_dir=str(out_dir / "checkpoints"),
        )
        res = execute_simulation_job(job)
        assert res.status == "COMPLETED"
        # Extract dark count metric or total clicks
        d_cnt = res.metrics.get("dark_counts", 0)
        dark_counts.append(d_cnt)
        print(f"  Temp={t_val} C -> Dark Counts={d_cnt}")

    # Physical expectation: higher temperature increases thermal dark counts
    passed = dark_counts[-1] >= dark_counts[0]
    print(f"  Dark counts across temperatures: {dark_counts}")
    print(f"  Experiment 3: {'PASSED' if passed else 'FAILED'}")
    return {
        "experiment": "3_environmental_variation",
        "passed": passed,
        "temperatures": temperatures,
        "dark_counts": dark_counts,
    }


def exp4_infrastructure_variation() -> dict[str, Any]:
    print("=" * 70)
    print("EXPERIMENT 4: Infrastructure Variation -> Physical Scaling")
    print("=" * 70)

    base = default_config()
    pulses = 10000
    distances = [10.0, 25.0, 45.0]
    out_dir = pathlib.Path("data/v05_experiments/exp4")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    signal_detections = []

    for dist in distances:
        overrides = {
            "protocol.n_pulses": pulses,
            "telemetry.window_size": 2500,
            "channel.distance_km": dist,
            "channel.attenuation_db_per_km": 0.20,
            "channel.insertion_loss_db": 1.0,
            "attack.enabled": False,
        }
        cfg = apply_overrides(base, overrides)
        scen = Scenario(
            scenario_id=f"scen_dist_{int(dist)}",
            infrastructure_id=f"dev_dist_{int(dist)}",
            seed=7777,
            config=cfg,
        )
        job = SimulationJob(
            scenario=scen,
            run_id="run_000",
            seed=derive_run_seed(scen.seed, "run_000"),
            output_dir=str(out_dir),
            checkpoint_dir=str(out_dir / "checkpoints"),
        )
        res = execute_simulation_job(job)
        assert res.status == "COMPLETED"
        raw_det = res.metrics.get("raw_detections", 0)
        signal_detections.append(raw_det)
        print(f"  Distance={dist} km -> Raw detections={raw_det}")

    # Physical scaling: detection count decays exponentially with distance (Beer-Lambert law)
    passed = signal_detections[0] > signal_detections[1] > signal_detections[2]
    print(f"  Detections with distance: {signal_detections}")
    print(f"  Experiment 4: {'PASSED' if passed else 'FAILED'}")
    return {
        "experiment": "4_infrastructure_variation",
        "passed": passed,
        "distances": distances,
        "signal_detections": signal_detections,
    }


def exp5_attack_variation() -> dict[str, Any]:
    print("=" * 70)
    print("EXPERIMENT 5: Attack Variation -> Distinct Observability Signatures")
    print("=" * 70)

    base = default_config()
    pulses = 10000
    out_dir = pathlib.Path("data/v05_experiments/exp5")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    attack_cases = [
        ("none", 0.0),
        ("intercept_resend", 0.8),
        ("channel_loss", 0.7),
    ]

    metrics_by_case = {}

    for atk_type, sev in attack_cases:
        overrides: dict[str, Any] = {
            "protocol.n_pulses": pulses,
            "telemetry.window_size": 2500,
        }
        if atk_type == "none":
            overrides["attack.enabled"] = False
            overrides["attack.scenarios"] = []
        else:
            overrides["attack.enabled"] = True
            overrides["attack.scenarios"] = [
                AttackScenarioConfig(
                    scenario_id=f"atk_{atk_type}",
                    type=atk_type,
                    severity=sev,
                    start_pulse=2000,
                    end_pulse=8000,
                    onset_pulses=500,
                    termination_pulses=500,
                    params={"fraction": 1.0} if atk_type == "intercept_resend" else {"excess_loss_db": 6.0},
                ).model_dump()
            ]

        cfg = apply_overrides(base, overrides)
        scen = Scenario(
            scenario_id=f"scen_atk_{atk_type}",
            infrastructure_id="dev_atk_test",
            seed=8888,
            config=cfg,
        )
        job = SimulationJob(
            scenario=scen,
            run_id="run_000",
            seed=derive_run_seed(scen.seed, "run_000"),
            output_dir=str(out_dir),
            checkpoint_dir=str(out_dir / "checkpoints"),
        )
        res = execute_simulation_job(job)
        assert res.status == "COMPLETED"
        qber = res.metrics.get("sifted_qber", 0.0)
        raw_det = res.metrics.get("raw_detections", 0)
        metrics_by_case[atk_type] = {"qber": qber, "raw_detections": raw_det}
        print(f"  Attack={atk_type:18s} -> QBER={qber:.4f}, detections={raw_det}")

    # Intercept-resend should elevate QBER (~0.20-0.25)
    # Channel loss should reduce raw detection counts relative to baseline
    qber_ir = metrics_by_case["intercept_resend"]["qber"]
    qber_none = metrics_by_case["none"]["qber"]
    det_loss = metrics_by_case["channel_loss"]["raw_detections"]
    det_none = metrics_by_case["none"]["raw_detections"]

    passed = (qber_ir > qber_none + 0.10) and (det_loss < det_none)
    print(f"  Experiment 5: {'PASSED' if passed else 'FAILED'}")
    return {
        "experiment": "5_attack_variation",
        "passed": passed,
        "metrics": metrics_by_case,
    }


def exp6_combined_variation() -> dict[str, Any]:
    print("=" * 70)
    print("EXPERIMENT 6: Combined Variation -> Population Diversity & Clean Provenance")
    print("=" * 70)

    pop_cfg = PopulationConfig(
        channel={
            "distance_km": ParameterDistributionConfig(
                distribution="uniform", a=10.0, b=35.0, min_val=5.0, max_val=45.0, unit="km"
            )
        },
        detectors={
            "efficiency": ParameterDistributionConfig(
                distribution="truncated_normal", mean=0.16, std=0.025, min_val=0.08, max_val=0.25
            )
        },
        source={
            "mu_signal": ParameterDistributionConfig(
                distribution="truncated_normal", mean=0.50, std=0.04, min_val=0.30, max_val=0.65
            )
        },
    )
    generator = ScenarioGenerator(pop_cfg, master_seed=4321)
    split_cfg = SplitConfig(
        n_scenarios=5,
        runs_per_scenario=1,
        pulses_per_run=10000,
        edge_case_fraction=0.20,
        attack_fraction=0.60,
        attack_types=["intercept_resend", "channel_loss"],
    )
    scenarios = generator.generate_split_scenarios("combined_test", split_cfg, split_seed=9876)

    out_dir = pathlib.Path("data/v05_experiments/exp6")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    scen_details = []
    for scen in scenarios:
        run_seed = derive_run_seed(scen.seed, "run_000")
        job = SimulationJob(
            scenario=scen,
            run_id="run_000",
            seed=run_seed,
            output_dir=str(out_dir),
            checkpoint_dir=str(out_dir / "checkpoints"),
        )
        res = execute_simulation_job(job)
        assert res.status == "COMPLETED"
        scen_details.append({
            "scenario_id": scen.scenario_id,
            "regime": scen.regime,
            "is_edge_case": scen.is_edge_case,
            "attack_enabled": scen.config.attack.enabled,
            "attack_type": scen.config.attack.type if scen.config.attack.enabled else "none",
            "qber": res.metrics.get("sifted_qber", 0.0),
            "detections": res.metrics.get("raw_detections", 0),
        })
        print(f"  Scen {scen.scenario_id} [{scen.regime}]: atk={scen_details[-1]['attack_type']:18s} det={scen_details[-1]['detections']} qber={scen_details[-1]['qber']:.4f}")

    # Diverse regimes, both normal and attack runs present
    atks_present = any(s["attack_enabled"] for s in scen_details)
    normals_present = any(not s["attack_enabled"] for s in scen_details)
    passed = atks_present and normals_present and len(scen_details) == 5
    print(f"  Experiment 6: {'PASSED' if passed else 'FAILED'}")
    return {
        "experiment": "6_combined_variation",
        "passed": passed,
        "scenarios": scen_details,
    }


def main():
    print("\nStarting QISM V0.5 Controlled Experiments Suite...\n")
    results = {}
    results["exp1"] = exp1_fixed_configuration()
    results["exp2"] = exp2_population_variation()
    results["exp3"] = exp3_environmental_variation()
    results["exp4"] = exp4_infrastructure_variation()
    results["exp5"] = exp5_attack_variation()
    results["exp6"] = exp6_combined_variation()

    all_passed = all(r["passed"] for r in results.values())
    print("\n" + "=" * 70)
    print(f"EXPERIMENTS SUMMARY: {'ALL 6 PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {v['experiment']}: {'PASSED' if v['passed'] else 'FAILED'}")

    report_path = pathlib.Path("reports/v05_experiments_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved experiments summary report to {report_path}\n")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
