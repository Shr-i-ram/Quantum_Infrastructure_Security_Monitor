"""Statistical parameter distribution abstractions for QISM population modeling.

In accordance with Requirements 12-16:
- Every randomized parameter has an explicit, scientifically meaningful distribution.
- Uniform distribution is NOT blindly applied to all parameters.
- Truncated normal, log-normal, beta, and normal distributions model hardware/environmental variations.
- Each distribution carries metadata: name, unit, physical bounds, reasoning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.stats import beta as sp_beta, truncnorm

from qism.config import ParameterDistributionConfig
from qism.types import DistributionType


@dataclass(frozen=True)
class ParameterMetadata:
    """Metadata describing the physical meaning, unit, and bounds of a parameter."""

    name: str
    distribution_type: str
    unit: str
    bounds: tuple[float | None, float | None]
    description: str


class SampledParameter:
    """Represents a configurable parameter with an explicit probability distribution."""

    def __init__(self, metadata: ParameterMetadata):
        self.metadata = metadata

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def unit(self) -> str:
        return self.metadata.unit

    def sample(self, rng: np.random.Generator) -> Any:
        """Sample a value from this distribution using the provided deterministic RNG."""
        raise NotImplementedError

    def cdf(self, x: float) -> float:
        """Cumulative distribution function (for statistical goodness-of-fit tests)."""
        raise NotImplementedError

    def ppf(self, q: float) -> float:
        """Percent-point function / inverse CDF (for Latin Hypercube Sampling)."""
        raise NotImplementedError


class FixedDistribution(SampledParameter):
    """Deterministic constant parameter value."""

    def __init__(self, name: str, value: Any, unit: str = "dimensionless", description: str = ""):
        super().__init__(ParameterMetadata(name, "fixed", unit, (None, None), description))
        self.value = value

    def sample(self, rng: np.random.Generator) -> Any:
        return self.value

    def cdf(self, x: float) -> float:
        return 1.0 if x >= self.value else 0.0

    def ppf(self, q: float) -> float:
        return float(self.value)


class UniformDistribution(SampledParameter):
    """Uniform continuous distribution U(a, b)."""

    def __init__(self, name: str, a: float, b: float, unit: str = "dimensionless", description: str = ""):
        if b <= a:
            raise ValueError(f"Uniform distribution requires b > a, got a={a}, b={b}")
        super().__init__(ParameterMetadata(name, "uniform", unit, (a, b), description))
        self.a = float(a)
        self.b = float(b)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.a, self.b))

    def cdf(self, x: float) -> float:
        if x <= self.a:
            return 0.0
        if x >= self.b:
            return 1.0
        return (x - self.a) / (self.b - self.a)

    def ppf(self, q: float) -> float:
        q = np.clip(q, 0.0, 1.0)
        return float(self.a + q * (self.b - self.a))


class NormalDistribution(SampledParameter):
    """Gaussian distribution N(mu, sigma^2)."""

    def __init__(self, name: str, mean: float, std: float, unit: str = "dimensionless", description: str = ""):
        if std <= 0:
            raise ValueError(f"Normal distribution requires std > 0, got {std}")
        super().__init__(ParameterMetadata(name, "normal", unit, (None, None), description))
        self.mean = float(mean)
        self.std = float(std)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mean, self.std))

    def cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf((x - self.mean) / (self.std * math.sqrt(2.0))))

    def ppf(self, q: float) -> float:
        from scipy.stats import norm
        return float(norm.ppf(q, loc=self.mean, scale=self.std))


class TruncatedNormalDistribution(SampledParameter):
    """Truncated normal distribution TN(mu, sigma, min_val, max_val)."""

    def __init__(
        self,
        name: str,
        mean: float,
        std: float,
        min_val: float,
        max_val: float,
        unit: str = "dimensionless",
        description: str = "",
    ):
        if max_val <= min_val:
            raise ValueError(f"TruncatedNormal requires max_val > min_val, got min={min_val}, max={max_val}")
        if std <= 0:
            raise ValueError(f"TruncatedNormal requires std > 0, got {std}")
        super().__init__(ParameterMetadata(name, "truncated_normal", unit, (min_val, max_val), description))
        self.mean = float(mean)
        self.std = float(std)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.a_std = (min_val - mean) / std
        self.b_std = (max_val - mean) / std

    def sample(self, rng: np.random.Generator) -> float:
        # Use uniform inverse CDF for deterministic sampling with numpy Generator
        u = rng.uniform(0.0, 1.0)
        return self.ppf(u)

    def cdf(self, x: float) -> float:
        return float(truncnorm.cdf(x, self.a_std, self.b_std, loc=self.mean, scale=self.std))

    def ppf(self, q: float) -> float:
        return float(truncnorm.ppf(q, self.a_std, self.b_std, loc=self.mean, scale=self.std))


class LogNormalDistribution(SampledParameter):
    """Log-normal distribution for strictly positive multiplicative quantities."""

    def __init__(
        self,
        name: str,
        mean: float,
        std: float,
        min_val: float | None = None,
        max_val: float | None = None,
        unit: str = "dimensionless",
        description: str = "",
    ):
        if std <= 0:
            raise ValueError(f"LogNormal requires std > 0, got {std}")
        super().__init__(ParameterMetadata(name, "log_normal", unit, (min_val, max_val), description))
        self.mean = float(mean)  # mu parameter of log(X)
        self.std = float(std)    # sigma parameter of log(X)
        self.min_val = min_val
        self.max_val = max_val

    def sample(self, rng: np.random.Generator) -> float:
        val = float(rng.lognormal(self.mean, self.std))
        if self.min_val is not None and val < self.min_val:
            val = self.min_val
        if self.max_val is not None and val > self.max_val:
            val = self.max_val
        return val

    def cdf(self, x: float) -> float:
        from scipy.stats import lognorm
        return float(lognorm.cdf(x, s=self.std, scale=math.exp(self.mean)))

    def ppf(self, q: float) -> float:
        from scipy.stats import lognorm
        val = float(lognorm.ppf(q, s=self.std, scale=math.exp(self.mean)))
        if self.min_val is not None:
            val = max(val, self.min_val)
        if self.max_val is not None:
            val = min(val, self.max_val)
        return val


class BetaDistribution(SampledParameter):
    """Beta distribution Beta(alpha, beta) scaled to interval [min_val, max_val]."""

    def __init__(
        self,
        name: str,
        alpha: float,
        beta: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
        unit: str = "dimensionless",
        description: str = "",
    ):
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"Beta distribution requires alpha, beta > 0, got alpha={alpha}, beta={beta}")
        if max_val <= min_val:
            raise ValueError(f"Beta distribution requires max_val > min_val, got min={min_val}, max={max_val}")
        super().__init__(ParameterMetadata(name, "beta", unit, (min_val, max_val), description))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def sample(self, rng: np.random.Generator) -> float:
        b = float(rng.beta(self.alpha, self.beta))
        return self.min_val + b * (self.max_val - self.min_val)

    def cdf(self, x: float) -> float:
        if x <= self.min_val:
            return 0.0
        if x >= self.max_val:
            return 1.0
        scaled = (x - self.min_val) / (self.max_val - self.min_val)
        return float(sp_beta.cdf(scaled, self.alpha, self.beta))

    def ppf(self, q: float) -> float:
        scaled = float(sp_beta.ppf(q, self.alpha, self.beta))
        return self.min_val + scaled * (self.max_val - self.min_val)


class CategoricalDistribution(SampledParameter):
    """Discrete categorical distribution over choices with optional weights."""

    def __init__(
        self,
        name: str,
        choices: list[Any],
        weights: list[float] | None = None,
        unit: str = "category",
        description: str = "",
    ):
        if not choices:
            raise ValueError("Categorical distribution requires non-empty choices")
        super().__init__(ParameterMetadata(name, "categorical", unit, (None, None), description))
        self.choices = list(choices)
        if weights is not None:
            if len(weights) != len(choices):
                raise ValueError("Weights length must match choices length")
            tot = sum(weights)
            self.probs = [w / tot for w in weights]
        else:
            self.probs = [1.0 / len(choices)] * len(choices)

    def sample(self, rng: np.random.Generator) -> Any:
        idx = rng.choice(len(self.choices), p=self.probs)
        return self.choices[idx]

    def cdf(self, x: float) -> float:
        raise NotImplementedError("CDF is not defined for categorical choices")

    def ppf(self, q: float) -> Any:
        cum = 0.0
        for choice, prob in zip(self.choices, self.probs):
            cum += prob
            if q <= cum:
                return choice
        return self.choices[-1]


def build_distribution(name: str, config: ParameterDistributionConfig) -> SampledParameter:
    """Factory creating a SampledParameter from a configuration object."""
    dist = str(config.distribution).lower()
    unit = config.unit
    desc = config.description

    if dist == "fixed":
        return FixedDistribution(name, config.value, unit=unit, description=desc)

    if dist == "uniform":
        a = config.a if config.a is not None else (config.min_val if config.min_val is not None else 0.0)
        b = config.b if config.b is not None else (config.max_val if config.max_val is not None else 1.0)
        return UniformDistribution(name, a, b, unit=unit, description=desc)

    if dist == "normal":
        mean = config.mean if config.mean is not None else 0.0
        std = config.std if config.std is not None else 1.0
        return NormalDistribution(name, mean, std, unit=unit, description=desc)

    if dist == "truncated_normal":
        mean = config.mean if config.mean is not None else 0.5
        std = config.std if config.std is not None else 0.1
        min_v = config.min_val if config.min_val is not None else 0.0
        max_v = config.max_val if config.max_val is not None else 1.0
        return TruncatedNormalDistribution(name, mean, std, min_v, max_v, unit=unit, description=desc)

    if dist == "log_normal":
        mean = config.mean if config.mean is not None else 0.0
        std = config.std if config.std is not None else 0.5
        return LogNormalDistribution(name, mean, std, config.min_val, config.max_val, unit=unit, description=desc)

    if dist == "beta":
        alpha = config.alpha if config.alpha is not None else 2.0
        beta = config.beta if config.beta is not None else 2.0
        min_v = config.min_val if config.min_val is not None else 0.0
        max_v = config.max_val if config.max_val is not None else 1.0
        return BetaDistribution(name, alpha, beta, min_v, max_v, unit=unit, description=desc)

    if dist == "categorical":
        choices = config.choices or []
        weights = config.weights
        return CategoricalDistribution(name, choices, weights, unit=unit, description=desc)

    raise ValueError(f"Unknown parameter distribution: {config.distribution!r}")
