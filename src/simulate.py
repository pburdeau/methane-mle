"""
Simulate a two-state intermittent methane emission process with the
joint observation model (snapshot + continuous monitoring).

Hybrid observation model:
  - Snapshot: snap_mask[t] ~ Bernoulli(p_snap) independently at each t.
  - Continuous: at each unmonitored step, Bernoulli(p_cont) triggers
    deployment of a monitor that runs for T_cont contiguous steps.
    During an active window no new deployment draw is made.
Both can be active at the same time step.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .utils import SiteObservations, TechnologySpec


# ═══════════════════════════════════════════════════════════════════
# Observation mask generation
# ═══════════════════════════════════════════════════════════════════

def generate_masks(
    T: int,
    p_snap: float,
    p_cont: float,
    T_cont: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate observation masks (hybrid model).

    Snapshot: independent Bernoulli(p_snap) at each time step.
    Continuous: at each unmonitored step, Bernoulli(p_cont) triggers
    deployment of a monitor that runs for T_cont contiguous steps.
    During an active window no new deployment draw is made.
    Snapshots can still occur during a continuous window.

    Returns
    -------
    snap_mask : ndarray, shape (T,), dtype bool
    cont_mask : ndarray, shape (T,), dtype bool
    """
    snap_mask = rng.random(T) < p_snap
    cont_mask = np.zeros(T, dtype=bool)
    t = 0
    while t < T:
        if rng.random() < p_cont:
            end = min(t + T_cont, T)
            cont_mask[t:end] = True
            t = end
        else:
            t += 1
    return snap_mask, cont_mask


# ═══════════════════════════════════════════════════════════════════
# Main simulation
# ═══════════════════════════════════════════════════════════════════

def simulate_series(
    T: int,
    p_on: float,
    p_off: float,
    size_mu: float,
    size_sigma: float,
    tech_specs: Sequence[TechnologySpec],
    snap_mask: np.ndarray,
    cont_mask: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[SiteObservations, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate emissions and observations under the joint observation model.

    Parameters
    ----------
    tech_specs : list of TechnologySpec
        Index 0 = snapshot, index 1 = continuous.
    snap_mask, cont_mask : bool arrays of shape (T,)
        Which time steps each technology observes.

    Returns
    -------
    obs : SiteObservations
    states : ndarray, shape (T,) — 0/1 hidden state.
    true_sizes : ndarray, shape (T,) — emission size (0 when OFF).
    event_ids : ndarray, shape (T,), dtype int — ON-period index (-1 when OFF).
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    rng = np.random.default_rng(rng)

    # ── Hidden Markov chain ───────────────────────────────────────
    states = np.zeros(T, dtype=int)
    pi_on = p_on / max(p_on + p_off, 1e-12)
    states[0] = 1 if rng.random() < pi_on else 0

    for t in range(1, T):
        if states[t - 1] == 0:
            states[t] = 1 if rng.random() < p_on else 0
        else:
            states[t] = 0 if rng.random() < p_off else 1

    # ── Emission sizes (constant within each ON period) ───────────
    true_sizes = np.zeros(T, dtype=float)
    event_ids = np.full(T, -1, dtype=int)
    current_size = 0.0
    current_event = -1

    for t in range(T):
        if states[t] == 1:
            if t == 0 or states[t - 1] == 0:
                current_size = float(rng.lognormal(mean=size_mu, sigma=size_sigma))
                current_event += 1
            true_sizes[t] = current_size
            event_ids[t] = current_event
        else:
            current_size = 0.0
            true_sizes[t] = 0.0

    # ── Detection and measurement (per technology) ────────────────
    snap_detected = np.zeros(T, dtype=bool)
    snap_measurements = np.full(T, np.nan, dtype=float)
    cont_detected = np.zeros(T, dtype=bool)
    cont_measurements = np.full(T, np.nan, dtype=float)

    for t in range(T):
        # Snapshot channel
        if snap_mask[t]:
            spec = tech_specs[0]
            if states[t] == 1:
                size = true_sizes[t]
                pod = float(spec.pod.probability(size))
                if rng.random() < pod:
                    snap_detected[t] = True
                    snap_measurements[t] = (
                        size if spec.sensor is None
                        else spec.sensor.sample(size, rng)
                    )
            else:
                if spec.false_positive_rate > 0 and rng.random() < spec.false_positive_rate:
                    snap_detected[t] = True
                    fp_size = spec.false_positive_scale
                    snap_measurements[t] = (
                        fp_size if spec.sensor is None
                        else spec.sensor.sample(fp_size, rng)
                    )

        # Continuous channel
        if cont_mask[t]:
            spec = tech_specs[1]
            if states[t] == 1:
                size = true_sizes[t]
                pod = float(spec.pod.probability(size))
                if rng.random() < pod:
                    cont_detected[t] = True
                    cont_measurements[t] = (
                        size if spec.sensor is None
                        else spec.sensor.sample(size, rng)
                    )
            else:
                if spec.false_positive_rate > 0 and rng.random() < spec.false_positive_rate:
                    cont_detected[t] = True
                    fp_size = spec.false_positive_scale
                    cont_measurements[t] = (
                        fp_size if spec.sensor is None
                        else spec.sensor.sample(fp_size, rng)
                    )

    obs = SiteObservations(
        T=T,
        snap_mask=snap_mask,
        snap_detected=snap_detected,
        snap_measurements=snap_measurements,
        cont_mask=cont_mask,
        cont_detected=cont_detected,
        cont_measurements=cont_measurements,
    )
    return obs, states, true_sizes, event_ids
