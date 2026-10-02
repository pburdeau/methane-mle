"""Core library for methane MLE estimation (joint observation model)."""

from .utils import (
    TechnologySpec,
    LogisticPOD,
    LognormalSensor,
    PiecewiseSensor,
    SiteObservations,
    lognormal_expectation,
    build_baseline_specs,
    build_perfect_specs,
    compute_threshold,
)
from .simulate import simulate_series, generate_masks
from .estimate import (
    identify_plumes,
    loop_empirical,
    Plume,
)
from .baselines import naive_baseline, pod_weighted_baseline, mle_simple, mle_simple_unbiased
from .estimate_variants import (
    loop_mle_constrained,
    loop_mle_plume_kappa,
    loop_mle_penalized,
)

__all__ = [
    "TechnologySpec",
    "LogisticPOD",
    "LognormalSensor",
    "PiecewiseSensor",
    "SiteObservations",
    "lognormal_expectation",
    "build_baseline_specs",
    "build_perfect_specs",
    "compute_threshold",
    "simulate_series",
    "generate_masks",
    "identify_plumes",
    "loop_empirical",
    "Plume",
    "naive_baseline",
    "pod_weighted_baseline",
    "loop_mle_constrained",
    "loop_mle_plume_kappa",
    "loop_mle_penalized",
]
