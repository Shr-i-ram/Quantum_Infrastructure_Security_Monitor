"""Population modeling, parameter distributions, and scenario generation package."""

from qism.population.dependencies import resolve_dependencies
from qism.population.distributions import (
    BetaDistribution,
    CategoricalDistribution,
    FixedDistribution,
    LogNormalDistribution,
    NormalDistribution,
    SampledParameter,
    TruncatedNormalDistribution,
    UniformDistribution,
    build_distribution,
)
from qism.population.generator import ScenarioGenerator, compute_config_hash
from qism.population.sampling import latin_hypercube_sample, monte_carlo_sample, sample_parameter_population
from qism.population.validator import validate_scenario_config

__all__ = [
    "BetaDistribution",
    "CategoricalDistribution",
    "FixedDistribution",
    "LogNormalDistribution",
    "NormalDistribution",
    "SampledParameter",
    "ScenarioGenerator",
    "TruncatedNormalDistribution",
    "UniformDistribution",
    "build_distribution",
    "compute_config_hash",
    "latin_hypercube_sample",
    "monte_carlo_sample",
    "resolve_dependencies",
    "sample_parameter_population",
    "validate_scenario_config",
]
