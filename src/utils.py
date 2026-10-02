"""
Data structures, POD models, sensor models, and shared utilities.

Key change from the old codebase: ObservationSeries is replaced by
SiteObservations, which stores per-technology observation arrays
(snapshot and continuous channels) rather than a single merged array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt, pi
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss


ArrayLike = np.ndarray | float


# ═══════════════════════════════════════════════════════════════════
# Lognormal helpers
# ═══════════════════════════════════════════════════════════════════

def lognormal_pdf(x: ArrayLike, mu: float, sigma: float) -> np.ndarray:
    """Evaluate the lognormal density at *x*."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-12, None)
    z = (np.log(x) - mu) / sigma
    return (np.exp(-0.5 * z**2) / (x * sigma * sqrt(2 * pi))).astype(float)


def lognormal_expectation(
    mu: float,
    sigma: float,
    func: Callable[[np.ndarray], np.ndarray],
    quad_points: int = 60,
) -> float:
    """E[func(S)] where S ~ LogNormal(mu, sigma), via Gauss-Hermite quadrature."""
    nodes, weights = hermgauss(quad_points)
    z = np.sqrt(2.0) * sigma * nodes + mu
    samples = np.exp(z)
    values = func(samples)
    return float(np.sum(weights * values) / sqrt(pi))


# ═══════════════════════════════════════════════════════════════════
# POD models
# ═══════════════════════════════════════════════════════════════════

class PODModel:
    """Base class for probability-of-detection curves."""

    def probability(self, size: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, size: ArrayLike) -> np.ndarray:
        return self.probability(size)


@dataclass
class LogisticPOD(PODModel):
    """Logistic POD: q(e) = 1 / (1 + (theta/e)^k)."""

    threshold: float
    slope: float

    def probability(self, size: ArrayLike) -> np.ndarray:
        size = np.asarray(size, dtype=float)
        size = np.clip(size, 1e-12, None)
        z = (np.log(size) - np.log(self.threshold)) * self.slope
        return 1.0 / (1.0 + np.exp(-z))


# ═══════════════════════════════════════════════════════════════════
# Sensor models
# ═══════════════════════════════════════════════════════════════════

class SensorModel:
    """Base class for conditional measurement noise models."""

    def sample(self, true_size: float, rng: np.random.Generator) -> float:
        raise NotImplementedError

    def pdf(self, observed: ArrayLike, true_size: ArrayLike) -> np.ndarray:
        raise NotImplementedError


@dataclass
class LognormalSensor(SensorModel):
    """Multiplicative lognormal measurement noise: M = e * exp(N(bias, sigma^2))."""

    sigma: float
    bias: float = 0.0

    def sample(self, true_size: float, rng: np.random.Generator) -> float:
        if true_size <= 0:
            raise ValueError("true_size must be positive for lognormal noise.")
        noise = rng.normal(loc=self.bias, scale=self.sigma)
        return float(true_size * np.exp(noise))

    def pdf(self, observed: ArrayLike, true_size: ArrayLike) -> np.ndarray:
        observed = np.asarray(observed, dtype=float)
        true_size = np.asarray(true_size, dtype=float)
        observed = np.clip(observed, 1e-12, None)
        true_size = np.clip(true_size, 1e-12, None)
        mu = np.log(true_size) + self.bias
        z = (np.log(observed) - mu) / self.sigma
        return (np.exp(-0.5 * z**2) / (observed * self.sigma * sqrt(2 * pi))).astype(float)


@dataclass
class PiecewiseSensor(SensorModel):
    """Wigle et al. (2024) piecewise sensor model."""

    alpha0: float
    gamma: float
    beta1: float
    beta2: float
    sigma: float

    def _phi(self, Q: ArrayLike) -> np.ndarray:
        Q = np.asarray(Q, dtype=float)
        quad = self.alpha0 + self.beta2 * (Q**2)
        lin = self.beta1 * Q
        return np.where(Q < self.gamma, quad, lin)

    def sample(self, true_size: float, rng: np.random.Generator) -> float:
        median = float(self._phi(true_size))
        if median <= 0:
            median = 1e-12
        noise = rng.normal(loc=0.0, scale=self.sigma)
        return float(median * np.exp(noise))

    def pdf(self, observed: ArrayLike, true_size: ArrayLike) -> np.ndarray:
        observed = np.asarray(observed, dtype=float)
        true_size = np.asarray(true_size, dtype=float)
        median = np.clip(self._phi(true_size), 1e-12, None)
        observed = np.clip(observed, 1e-12, None)
        mu_log = np.log(median)
        z = (np.log(observed) - mu_log) / self.sigma
        return (np.exp(-0.5 * z**2) / (observed * self.sigma * sqrt(2 * pi))).astype(float)


# ═══════════════════════════════════════════════════════════════════
# Technology specification
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TechnologySpec:
    """Monitoring technology with POD curve, sensor model, and false-positive behaviour."""

    pod: PODModel
    sensor: Optional[SensorModel] = None
    false_positive_rate: float = 0.0
    false_positive_scale: float = 1.0


# ═══════════════════════════════════════════════════════════════════
# Observation data structure — dual-channel (snapshot + continuous)
# ═══════════════════════════════════════════════════════════════════

class SiteObservations:
    """Per-technology observation arrays for a single site over T time steps.

    Each technology has its own mask/detected/measurement arrays:
      - snap_mask[t]: True if snapshot tech observes at time t.
      - cont_mask[t]: True if continuous tech observes at time t.
    """

    def __init__(
        self,
        T: int,
        snap_mask: np.ndarray,
        snap_detected: np.ndarray,
        snap_measurements: np.ndarray,
        cont_mask: np.ndarray,
        cont_detected: np.ndarray,
        cont_measurements: np.ndarray,
    ) -> None:
        self._T = T
        self._snap_mask = np.asarray(snap_mask, dtype=bool)
        self._snap_detected = np.asarray(snap_detected, dtype=bool)
        self._snap_measurements = np.asarray(snap_measurements, dtype=float)
        self._cont_mask = np.asarray(cont_mask, dtype=bool)
        self._cont_detected = np.asarray(cont_detected, dtype=bool)
        self._cont_measurements = np.asarray(cont_measurements, dtype=float)

        for arr in (self._snap_mask, self._snap_detected, self._snap_measurements,
                    self._cont_mask, self._cont_detected, self._cont_measurements):
            if arr.shape != (T,):
                raise ValueError(f"All arrays must have shape ({T},), got {arr.shape}.")

    @property
    def T(self) -> int:
        return self._T

    @property
    def snap_mask(self) -> np.ndarray:
        return self._snap_mask

    @property
    def snap_detected(self) -> np.ndarray:
        return self._snap_detected

    @property
    def snap_measurements(self) -> np.ndarray:
        return self._snap_measurements

    @property
    def cont_mask(self) -> np.ndarray:
        return self._cont_mask

    @property
    def cont_detected(self) -> np.ndarray:
        return self._cont_detected

    @property
    def cont_measurements(self) -> np.ndarray:
        return self._cont_measurements

    # ── Derived properties ────────────────────────────────────────

    @property
    def any_observed(self) -> np.ndarray:
        """Boolean mask: any technology observing at each time step."""
        return self._snap_mask | self._cont_mask

    @property
    def any_detected(self) -> np.ndarray:
        """Boolean mask: any detection at each time step."""
        return ((self._snap_mask & self._snap_detected)
                | (self._cont_mask & self._cont_detected))

    def observed_times(self) -> np.ndarray:
        """Sorted array of time indices where at least one tech observes."""
        return np.flatnonzero(self.any_observed)

    def n_total_observations(self) -> int:
        """Total observation count (each tech counted separately)."""
        return int(np.sum(self._snap_mask) + np.sum(self._cont_mask))

    def n_total_detections(self) -> int:
        """Total detection count (each tech counted separately)."""
        snap_det = int(np.sum(self._snap_mask & self._snap_detected))
        cont_det = int(np.sum(self._cont_mask & self._cont_detected))
        return snap_det + cont_det

    def detection_list(self) -> list[tuple[int, int, float]]:
        """List of (time, tech_id, measurement) for all detections, sorted by time.

        tech_id: 0 = snapshot, 1 = continuous.
        """
        entries: list[tuple[int, int, float]] = []
        for t in range(self._T):
            if self._snap_mask[t] and self._snap_detected[t]:
                entries.append((t, 0, float(self._snap_measurements[t])))
            if self._cont_mask[t] and self._cont_detected[t]:
                entries.append((t, 1, float(self._cont_measurements[t])))
        return entries


# ═══════════════════════════════════════════════════════════════════
# Tech-spec factory helpers
# ═══════════════════════════════════════════════════════════════════

def build_baseline_specs(
    aerial_threshold: float,
    aerial_slope: float,
    aerial_sensor_sigma: float,
    cont_threshold: float,
    cont_slope: float,
    cont_sensor_sigma: float,
    cont_fp_rate: float = 0.005,
    cont_fp_scale: float = 4.0,
) -> list[TechnologySpec]:
    """Build the two-technology spec list used throughout the paper.

    Index 0 = snapshot (aerial), index 1 = continuous.
    """
    aerial_sensor = LognormalSensor(
        sigma=aerial_sensor_sigma,
        bias=-0.5 * aerial_sensor_sigma**2,
    )
    cont_sensor = LognormalSensor(
        sigma=cont_sensor_sigma,
        bias=-0.5 * cont_sensor_sigma**2,
    )
    return [
        TechnologySpec(
            pod=LogisticPOD(threshold=aerial_threshold, slope=aerial_slope),
            sensor=aerial_sensor,
            false_positive_rate=0.0,
        ),
        TechnologySpec(
            pod=LogisticPOD(threshold=cont_threshold, slope=cont_slope),
            sensor=cont_sensor,
            false_positive_rate=cont_fp_rate,
            false_positive_scale=cont_fp_scale,
        ),
    ]


def build_perfect_specs(
    aerial_slope: float = 2.0,
    aerial_sensor_sigma: float = 0.05,
    cont_slope: float = 3.0,
    cont_sensor_sigma: float = 0.2,
    cont_fp_rate: float = 0.005,
    cont_fp_scale: float = 4.0,
) -> list[TechnologySpec]:
    """Perfect detection (POD ~ 1 for all sizes): threshold -> 0."""
    return build_baseline_specs(
        aerial_threshold=0.001,
        aerial_slope=aerial_slope,
        aerial_sensor_sigma=aerial_sensor_sigma,
        cont_threshold=0.001,
        cont_slope=cont_slope,
        cont_sensor_sigma=cont_sensor_sigma,
        cont_fp_rate=cont_fp_rate,
        cont_fp_scale=cont_fp_scale,
    )


def compute_threshold(target_size: float, target_pod: float, slope: float) -> float:
    """Compute POD threshold from target detection rate at a given size."""
    ratio = (1.0 / target_pod) - 1.0
    return float(target_size * (ratio ** (1.0 / slope)))
