"""Iterative plug-in estimation with linked plumes and event-level IPW.

Steps 1 and 2 retain the original linking, measurement combination, and IPW.
Step 3 maximizes a persistent-size detection likelihood conditional on the
weighted empirical event-size distribution. Sizes persist across ON steps;
new events draw new sizes. This is not joint MLE or exact EM. No likelihood
nudge is applied. Fixed optimization bounds do not depend on generating truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .utils import SiteObservations, TechnologySpec
from .persistent import detection_inputs, fit_transitions, event_detection_probability


# ═══════════════════════════════════════════════════════════════════
# Plume data structure
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Plume:
    """A detected plume (group of linked observations)."""

    indices: List[int]
    measurements: List[float]
    tech_ids: List[int]

    @property
    def mean(self) -> float:
        return float(np.mean(self.measurements))

    @property
    def duration(self) -> int:
        return self.indices[-1] - self.indices[0] + 1


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _sensor_noise(spec: TechnologySpec, override: Optional[float] = None) -> float:
    """Return log-scale sigma for a tech spec."""
    if override is not None:
        return float(override)
    if spec.sensor is not None and hasattr(spec.sensor, "sigma"):
        return float(spec.sensor.sigma)
    return 0.0


def _combined_log_estimate(
    measurements: Sequence[float],
    tech_ids: Sequence[int],
    tech_specs: Sequence[TechnologySpec],
    noise_override: Optional[dict] = None,
) -> tuple[float, float]:
    """Inverse-variance weighted average in log-space.

    Returns (combined_measurement, effective_sigma).
    """
    log_ms: list[float] = []
    inv_vars: list[float] = []
    for meas, tech_id in zip(measurements, tech_ids):
        ov = noise_override.get(tech_id) if noise_override else None
        sigma = max(_sensor_noise(tech_specs[tech_id], ov), 1e-6)
        log_ms.append(np.log(max(meas, 1e-12)))
        inv_vars.append(1.0 / sigma ** 2)
    total_inv_var = sum(inv_vars)
    combined_log = sum(l * v for l, v in zip(log_ms, inv_vars)) / total_inv_var
    effective_sigma = 1.0 / np.sqrt(total_inv_var)
    return float(np.exp(combined_log)), float(effective_sigma)


# ═══════════════════════════════════════════════════════════════════
# Step 1: Bayesian plume identification
# ═══════════════════════════════════════════════════════════════════

def identify_plumes(
    obs: SiteObservations,
    tech_specs: Sequence[TechnologySpec],
    p_off: Optional[float] = None,
    max_gap: int = 3,
    decision_threshold: float = 0.5,
    sigma_b: float = 0.69,
    sigma_w: float = 0.0,
    noise_override: Optional[dict] = None,
) -> List[Plume]:
    """
    Bayesian plume linking on the detection list from SiteObservations.

    Detections at the same time step are always grouped (same emission
    event). Their measurements are combined via inverse-variance weighted
    average in log-space BEFORE computing the Bayes factor for linking
    with adjacent time-step groups.
    """
    det_list = obs.detection_list()
    if len(det_list) == 0:
        return []

    tau = (1.0 / p_off) if (p_off is not None and p_off > 0) else float(max_gap)

    # ── Group detections by time step ─────────────────────────────
    groups: list[list[tuple[int, int, float]]] = []
    i = 0
    while i < len(det_list):
        t = det_list[i][0]
        group = [det_list[i]]
        j = i + 1
        while j < len(det_list) and det_list[j][0] == t:
            group.append(det_list[j])
            j += 1
        groups.append(group)
        i = j

    # ── Combined measurement + effective noise per group ──────────
    group_stats: list[tuple[float, float]] = []
    for group in groups:
        measurements = [e[2] for e in group]
        tech_ids = [e[1] for e in group]
        combined_meas, effective_sigma = _combined_log_estimate(
            measurements, tech_ids, tech_specs, noise_override,
        )
        group_stats.append((combined_meas, effective_sigma))

    # ── Bayesian linking between consecutive groups ───────────────
    plumes: list[Plume] = []
    cur_entries: list[tuple[int, int, float]] = list(groups[0])

    for i in range(1, len(groups)):
        t_curr = groups[i][0][0]
        t_prev = groups[i - 1][0][0]
        delta_t = t_curr - t_prev

        meas_curr, sigma_curr = group_stats[i]
        meas_prev, sigma_prev = group_stats[i - 1]

        d = np.log(max(meas_curr, 1e-12)) - np.log(max(meas_prev, 1e-12))
        eps_var = 1e-20
        V_same = sigma_curr ** 2 + sigma_prev ** 2 + sigma_w ** 2 + eps_var
        V_diff = sigma_curr ** 2 + sigma_prev ** 2 + 2 * sigma_b ** 2 + eps_var

        log_BF = (0.5 * np.log(V_diff / V_same)
                  - (d ** 2 / 2.0) * (1.0 / V_same - 1.0 / V_diff))
        exp_term = np.exp(-delta_t / tau)
        with np.errstate(divide="ignore"):  # underflow denotes negligible linking odds
            log_PO = np.log(exp_term) - np.log(1.0 - exp_term + 1e-12)
        log_odds = log_PO + log_BF

        if log_odds > 20:
            P_same = 1.0
        elif log_odds < -20:
            P_same = 0.0
        else:
            P_same = 1.0 / (1.0 + np.exp(-log_odds))

        if P_same > decision_threshold:
            cur_entries.extend(groups[i])
        else:
            plumes.append(Plume(
                [e[0] for e in cur_entries],
                [e[2] for e in cur_entries],
                [e[1] for e in cur_entries],
            ))
            cur_entries = list(groups[i])

    plumes.append(Plume(
        [e[0] for e in cur_entries],
        [e[2] for e in cur_entries],
        [e[1] for e in cur_entries],
    ))
    return plumes


# ═══════════════════════════════════════════════════════════════════
# One plug-in iteration (persistent event sizes)
# ═══════════════════════════════════════════════════════════════════

def _run_one_step(
    obs: SiteObservations,
    tech_specs: Sequence[TechnologySpec],
    p_on_in: float,
    p_off_in: float,
    max_gap: int = 3,
    decision_threshold: float = 0.5,
    transition_bounds: tuple = (1e-5, 1-1e-6),
    eps: float = 1e-10,
    noise_override: Optional[dict] = None,
    p_on_grid: Optional[np.ndarray] = None,
    p_off_grid: Optional[np.ndarray] = None,
    p_off_nudge: float = 1.0,
    coarse_search: bool = False,
    optimizer_tolerance: float = 1e-11,
) -> dict:
    """One iteration: plume linking, original event-IPW, conditional likelihood."""
    if p_off_nudge != 1.0:
        raise ValueError("The persistent-size estimator has no likelihood nudge; use 1.0.")
    plumes = identify_plumes(
        obs, tech_specs, p_off=p_off_in, max_gap=max_gap,
        decision_threshold=decision_threshold, noise_override=noise_override,
    )

    if len(plumes) == 0:
        return dict(
            p_on_new=p_on_in, p_off_new=p_off_in,
            plumes=[], plume_sizes=[], ipw_weights=[],
            weighted_mean=0.0, kappa_values=[0.0] * len(tech_specs),
            kappa_sc=0.0, log_likelihood=-np.inf,
        )

    # ── Plume sizes: inverse-variance weighted average in log-space ──
    # Correct for lognormal bias: exp(avg of logs) underestimates the
    # true emission size when n > 1 measurements are combined.
    # Correction factor: exp((n-1) / (2·W)) where W = Σ 1/σ_j².
    plume_sizes = np.empty(len(plumes))
    for m, p in enumerate(plumes):
        combined, eff_sigma = _combined_log_estimate(
            p.measurements, p.tech_ids, tech_specs, noise_override,
        )
        n_meas = len(p.measurements)
        if n_meas > 1 and eff_sigma > 0:
            total_inv_var = 1.0 / eff_sigma ** 2
            correction = (n_meas - 1) / (2.0 * total_inv_var)
            combined *= np.exp(correction)
        plume_sizes[m] = combined

    # ── Plume detection probabilities (marginal over event duration) ──
    # For an emission of size e, the probability of detection at least
    # once during the event is computed by marginalising over the event
    # duration D ~ Geometric(p_off) using the actual observation masks.
    #
    # Backward recursion:
    #   alpha[T] = 1  (beyond the monitoring window → guaranteed miss)
    #   alpha[t] = p_miss[t,e] * (p_off + (1-p_off)*alpha[t+1])
    #   P_det(e) = 1 - mean(alpha)
    #
    # Vectorised over a grid of emission sizes, then interpolated.
    n_grid = 60
    log_e_lo, log_e_hi = np.log(0.5), np.log(2000.0)
    log_e_grid = np.linspace(log_e_lo, log_e_hi, n_grid)
    e_grid = np.exp(log_e_grid)

    q_s_grid = tech_specs[0].pod.probability(e_grid)
    q_c_grid = tech_specs[1].pod.probability(e_grid)

    snap_2d = obs.snap_mask[:, np.newaxis].astype(float)
    cont_2d = obs.cont_mask[:, np.newaxis].astype(float)
    p_miss_2d = (1.0 - snap_2d * q_s_grid[np.newaxis, :]) * \
                (1.0 - cont_2d * q_c_grid[np.newaxis, :])

    p_det_grid = np.clip(event_detection_probability(p_off_in, p_miss_2d), eps, 1.0)

    plume_det_probs = np.empty(len(plumes))
    for m, _p in enumerate(plumes):
        log_e = np.log(max(plume_sizes[m], 0.5))
        log_e = np.clip(log_e, log_e_lo, log_e_hi)
        idx = (log_e - log_e_lo) / (log_e_hi - log_e_lo) * (n_grid - 1)
        i0 = int(idx)
        i1 = min(i0 + 1, n_grid - 1)
        frac = idx - i0
        plume_det_probs[m] = max(
            (1.0 - frac) * p_det_grid[i0] + frac * p_det_grid[i1], eps,
        )

    # ── IPW weights ──────────────────────────────────────────────
    weights = 1.0 / plume_det_probs
    mu_when_emitting = float(np.sum(weights * plume_sizes) / np.sum(weights))

    # ── κ estimation (including joint κ_sc) ──────────────────────
    q_s_vals = tech_specs[0].pod.probability(plume_sizes)
    q_c_vals = tech_specs[1].pod.probability(plume_sizes)
    w_sum = np.sum(weights)
    kappa_s = float(np.clip(np.sum(weights * q_s_vals) / w_sum, eps, 1 - eps))
    kappa_c = float(np.clip(np.sum(weights * q_c_vals) / w_sum, eps, 1 - eps))
    kappa_sc = float(np.clip(
        np.sum(weights * q_s_vals * q_c_vals) / w_sum, eps, 1 - eps,
    ))
    kappa_sc = min(kappa_sc, kappa_s, kappa_c)

    kappa_values = [kappa_s, kappa_c]

    # Retain the entire weighted empirical law, including persistent size.
    inputs = detection_inputs(obs, tech_specs, plume_sizes)
    (best_p_on, best_p_off), best_ll, optimizer_success = fit_transitions(
        weights, inputs, (p_on_in, p_off_in), transition_bounds,
        coarse=coarse_search, p_on_grid=p_on_grid, p_off_grid=p_off_grid,
        optimizer_tolerance=optimizer_tolerance,
    )

    return dict(
        p_on_new=best_p_on, p_off_new=best_p_off,
        plumes=plumes, plume_sizes=plume_sizes.tolist(),
        ipw_weights=weights.tolist(), weighted_mean=mu_when_emitting,
        kappa_values=kappa_values, kappa_sc=kappa_sc,
        log_likelihood=best_ll, optimizer_success=optimizer_success,
    )


# ═══════════════════════════════════════════════════════════════════
# Full loop estimator
# ═══════════════════════════════════════════════════════════════════

def loop_empirical(
    obs: SiteObservations,
    tech_specs: Sequence[TechnologySpec],
    T: int,
    max_gap: int = 3,
    init_p_on: float = 0.025,
    init_p_off: float = 0.10,
    max_iter: int = 30,
    tol: float = 1e-5,
    mean_tol: Optional[float] = None,
    transition_bounds: tuple = (1e-5, 1-1e-6),
    eps: float = 1e-10,
    noise_override: Optional[dict] = None,
    decision_threshold: float = 0.5,
    p_on_grid: Optional[np.ndarray] = None,
    p_off_grid: Optional[np.ndarray] = None,
    p_off_nudge: float = 1.0,
    optimizer_tolerance: float = 1e-11,
) -> dict:
    """
    Run the self-consistent loop estimator on dual-channel observations.

    Returns dict with keys: mean, p_on, p_off, mu_emit,
    empirical_distribution, converged, n_iterations.
    """
    if p_off_nudge != 1.0:
        raise ValueError("The persistent-size estimator has no likelihood nudge; use 1.0.")
    if T != obs.T:
        raise ValueError("T must match the observation campaign length.")
    if obs.n_total_detections() == 0:
        # Keep zero-detection campaigns in mean-performance summaries. Activity
        # and event size are unidentified; do not report fabricated parameters.
        return dict(mean=0.0 if obs.n_total_observations() else np.nan,
                    p_on=np.nan, p_off=np.nan, mu_emit=np.nan,
                    empirical_distribution=np.array([]), empirical_weights=np.array([]),
                    converged=False, n_iterations=0, optimizer_success=False,
                    has_detections=False, boundary=False, log_likelihood=np.nan)
    p_on, p_off = init_p_on, init_p_off
    converged = False
    n_iterations = 0
    prev_mean = None
    mean_tol_val = mean_tol if mean_tol is not None else 1e-4
    plume_sizes = np.array([])
    mu_emit = 0.0

    for iteration in range(max_iter):
        step = _run_one_step(
            obs, tech_specs, p_on, p_off,
            max_gap=max_gap, decision_threshold=decision_threshold,
            transition_bounds=transition_bounds, eps=eps,
            noise_override=noise_override,
            p_on_grid=p_on_grid, p_off_grid=p_off_grid,
            p_off_nudge=p_off_nudge, coarse_search=iteration == 0,
            optimizer_tolerance=optimizer_tolerance,
        )

        p_on_new = step["p_on_new"]
        p_off_new = step["p_off_new"]
        mu_emit = step["weighted_mean"]
        plume_sizes = np.array(step["plume_sizes"])

        pi_emit = p_on_new / (p_on_new + p_off_new)
        current_mean = mu_emit * pi_emit

        params_ok = abs(p_on_new - p_on) < tol and abs(p_off_new - p_off) < tol
        mean_ok = prev_mean is not None and abs(current_mean - prev_mean) < mean_tol_val
        converged = params_ok and mean_ok

        if converged:
            p_on, p_off = p_on_new, p_off_new
            n_iterations = iteration + 1
            break

        n_iterations = iteration + 1
        p_on, p_off = p_on_new, p_off_new
        prev_mean = current_mean

    if not converged:
        n_iterations = max_iter

    pi_emit = p_on / (p_on + p_off)
    mu_total = mu_emit * pi_emit

    return {
        "mean": mu_total,
        "p_on": p_on,
        "p_off": p_off,
        "mu_emit": mu_emit,
        "empirical_distribution": plume_sizes,
        "empirical_weights": np.array(step["ipw_weights"]),
        "optimizer_success": step["optimizer_success"],
        "has_detections": True,
        "boundary": bool(min(p_on,p_off) <= transition_bounds[0]*1.001 or
                         max(p_on,p_off) >= transition_bounds[1]*.9999),
        "log_likelihood": step["log_likelihood"],
        "converged": converged,
        "n_iterations": n_iterations,
    }
