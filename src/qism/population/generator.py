"""Scenario generator for population-based QKD simulation (Requirements 21, 22, 57–62).

Generates independent, frozen, physically consistent Scenario objects from
configured parameter distributions, dependency graphs, and split constraints.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Any
import numpy as np

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
from qism.population.dependencies import resolve_dependencies
from qism.population.distributions import SampledParameter, build_distribution
from qism.population.sampling import sample_parameter_population
from qism.population.validator import validate_scenario_config
from qism.random import derive_scenario_seed, make_rng

log = logging.getLogger("qism.population.generator")


def compute_config_hash(cfg: QKDConfig) -> str:
    """Compute a canonical, deterministic SHA-256 hash of a normalized QKDConfig."""
    data = cfg.model_dump(mode="json")
    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]


class ScenarioGenerator:
    """Generates population scenarios with explicit distributions and physical validation."""

    def __init__(
        self,
        population_config: PopulationConfig,
        base_template: QKDConfig | None = None,
        master_seed: int = 20260501,
        sampling_strategy: str = "latin_hypercube",
    ):
        self.population_config = population_config
        self.base_template = base_template or default_config()
        self.master_seed = master_seed
        self.sampling_strategy = sampling_strategy

        # Compile SampledParameter objects from population_config
        self._parameters: dict[str, SampledParameter] = {}
        self._compile_parameter_definitions()

    def _compile_parameter_definitions(self) -> None:
        """Flatten component taxonomy into a dictionary of SampledParameters."""
        sections = [
            ("source", self.population_config.source),
            ("channel", self.population_config.channel),
            ("detectors", self.population_config.detectors),
            ("timing", self.population_config.timing),
            ("environment", self.population_config.environment),
            ("attack", self.population_config.attack),
        ]
        for section_prefix, dist_dict in sections:
            for param_key, p_cfg in dist_dict.items():
                full_key = param_key if param_key.startswith(f"{section_prefix}.") else f"{section_prefix}.{param_key}"
                self._parameters[full_key] = build_distribution(full_key, p_cfg)

    def generate_split_scenarios(
        self,
        split_name: str,
        split_cfg: SplitConfig,
        split_seed: int,
    ) -> list[Scenario]:
        """Generate all validated Scenario instances for a dataset split."""
        n_scen = split_cfg.n_scenarios
        rng = make_rng(split_seed)

        # 1. Sample raw parameter combinations using configured strategy
        raw_samples = sample_parameter_population(
            self._parameters,
            n_scen,
            rng,
            strategy=self.sampling_strategy,
        )

        n_edge = int(round(n_scen * split_cfg.edge_case_fraction))
        edge_indices = set(rng.choice(n_scen, size=n_edge, replace=False)) if n_edge > 0 else set()

        scenarios: list[Scenario] = []

        for idx, raw_sample in enumerate(raw_samples):
            scen_id = f"{split_name}_{idx:04d}"
            infra_id = f"dev_{split_name}_{idx:04d}"
            scen_seed = derive_scenario_seed(split_seed, scen_id)
            scen_rng = make_rng(scen_seed)

            is_edge = idx in edge_indices

            # 2. Resolve parameter dependencies & physical correlations
            resolved_params, regime = resolve_dependencies(raw_sample, scen_rng, is_edge_case=is_edge)

            # 3. Apply split-specific logic (generalization holdouts, attacks, calibration)
            cfg_dict = copy.deepcopy(resolved_params)
            cfg_dict["protocol.n_pulses"] = split_cfg.pulses_per_run
            cfg_dict["simulation.scenario_id"] = scen_id

            if split_name == "generalization" and split_cfg.held_out_dimensions:
                # Apply held-out ranges
                for param_path, bounds in split_cfg.held_out_dimensions.items():
                    min_b = bounds.get("min", 0.0)
                    max_b = bounds.get("max", 1.0)
                    val = float(scen_rng.uniform(min_b, max_b))
                    if param_path == "detectors.efficiency":
                        cfg_dict["detectors.detector_0.efficiency"] = val
                        cfg_dict["detectors.detector_1.efficiency"] = val
                    elif param_path == "detectors.dark_count_rate_hz":
                        cfg_dict["detectors.detector_0.dark_count_rate_hz"] = val
                        cfg_dict["detectors.detector_1.dark_count_rate_hz"] = val
                    else:
                        cfg_dict[param_path] = val

            # Attack configuration
            if split_cfg.attack_fraction > 0:
                is_attack_run = scen_rng.random() < split_cfg.attack_fraction
                if is_attack_run:
                    cfg_dict["attack.enabled"] = True
                    atk_types = split_cfg.attack_types or ["intercept_resend"]
                    atk_type = str(scen_rng.choice(atk_types))
                    cfg_dict["attack.type"] = atk_type
                    
                    # Randomize attack timing and severity
                    pulses = split_cfg.pulses_per_run
                    # Start between 15% and 40% of run
                    start_p = int(scen_rng.uniform(0.15 * pulses, 0.40 * pulses))
                    # Duration between 25% and 50% of run
                    dur_p = int(scen_rng.uniform(0.25 * pulses, 0.50 * pulses))
                    end_p = min(pulses - 500, start_p + dur_p)
                    onset_p = int(scen_rng.uniform(500, 2000))
                    term_p = int(scen_rng.uniform(500, 2000))
                    sev = float(scen_rng.beta(2.0, 2.0)) # Bounded severity
                    
                    sc_cfg = AttackScenarioConfig(
                        scenario_id=f"atk_{scen_id}",
                        type=atk_type,
                        severity=sev,
                        start_pulse=start_p,
                        end_pulse=end_p,
                        onset_pulses=onset_p,
                        termination_pulses=term_p,
                        onset_profile="smoothstep",
                        recovery_profile="smoothstep",
                        params=self._sample_attack_params(atk_type, scen_rng),
                    )
                    cfg_dict["attack.scenarios"] = [sc_cfg.model_dump()]
                else:
                    cfg_dict["attack.enabled"] = False
                    cfg_dict["attack.scenarios"] = []
            else:
                cfg_dict["attack.enabled"] = False
                cfg_dict["attack.scenarios"] = []

            # 4. Build and validate QKDConfig
            candidate_cfg = apply_overrides(self.base_template, cfg_dict)
            errors = validate_scenario_config(candidate_cfg)
            if errors:
                log.warning("Scenario %s generated with validation errors: %s; repairing...", scen_id, errors)
                # Repair minor discrepancies if any
                candidate_cfg.channel.distance_km = max(1.0, candidate_cfg.channel.distance_km)
                candidate_cfg.detectors.detector_0.efficiency = float(np.clip(candidate_cfg.detectors.detector_0.efficiency, 0.05, 0.95))
                candidate_cfg.detectors.detector_1.efficiency = float(np.clip(candidate_cfg.detectors.detector_1.efficiency, 0.05, 0.95))
                remaining_errors = validate_scenario_config(candidate_cfg)
                if remaining_errors:
                    raise ValueError(f"Scenario {scen_id} failed physical validation: {remaining_errors}")

            # 5. Compute configuration hash
            c_hash = compute_config_hash(candidate_cfg)

            scenarios.append(
                Scenario(
                    scenario_id=scen_id,
                    infrastructure_id=infra_id,
                    seed=scen_seed,
                    regime=regime,
                    is_edge_case=is_edge,
                    dataset_split=split_name,
                    config=candidate_cfg,
                    sampled_parameters=resolved_params,
                    config_hash=c_hash,
                )
            )

        return scenarios

    def _sample_attack_params(self, atk_type: str, rng: np.random.Generator) -> dict[str, Any]:
        """Sample realistic attack-specific parameters."""
        if atk_type == "intercept_resend":
            return {"fraction": 1.0}
        if atk_type == "channel_loss":
            return {"delta_L_max_db": float(rng.uniform(4.0, 8.0))}
        if atk_type == "source_manipulation":
            dir_choice = str(rng.choice(["increase", "decrease"]))
            return {"delta_mu": float(rng.uniform(0.15, 0.40)), "direction": dir_choice}
        if atk_type == "detector_manipulation":
            mode_choice = str(rng.choice(["mismatch", "dark_count"]))
            if mode_choice == "mismatch":
                return {"mode": "mismatch", "delta_eta": float(rng.uniform(0.4, 0.8)), "target_detector": int(rng.choice([0, 1]))}
            return {"mode": "dark_count", "delta_dark_hz": float(rng.uniform(5000.0, 15000.0)), "target_detector": -1}
        if atk_type == "timing":
            return {"delta_t_max_s": float(rng.uniform(200.0e-12, 400.0e-12))}
        if atk_type == "pns":
            return {"single_photon_suppression": float(rng.uniform(0.5, 0.85))}
        return {}
