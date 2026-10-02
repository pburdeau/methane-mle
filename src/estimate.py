"""
Loop Empirical Estimator (MLE) with IPW correction.

Adapted for the joint observation model where snapshot and continuous
technologies observe independently at each time step.

Key mathematical properties implemented correctly:
  - κ_sc = E[q_s(Q)·q_c(Q)] computed as its own integral (NOT κ_s·κ_c).
  - Forward algorithm uses four-way dispatch for dual-deployment steps:
      both detect → κ_sc,  snap only → κ_s − κ_sc, etc.
  - IPW weight = 1/P(plume detected at ≥1 time step across its span).
  - Plume sizes combined via inverse-variance weighted average in log-space.

Steps per iteration:
  1. Identify plumes via Bayesian posterior linking (combine dual-channel
     measurements before computing Bayes factor).
  2. IPW-corrected empirical size distribution and expected POD
     (κ_s, κ_c, κ_sc).
  3. Grid search over (p_on, p_off) maximising the HMM forward-algorithm
     log-likelihood with joint emission probabilities.
  4. Repeat until convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .utils import SiteObservations, TechnologySpec


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
# Single EM iteration (joint observation model)
# ═══════════════════════════════════════════════════════════════════

def _run_one_step(
    obs: SiteObservations,
    tech_specs: Sequence[TechnologySpec],
    p_on_in: float,
    p_off_in: float,
    max_gap: int = 3,
    decision_threshold: float = 0.5,
    transition_bounds: tuple = (0.01, 0.95),
    eps: float = 1e-10,
    noise_override: Optional[dict] = None,
    p_on_grid: Optional[np.ndarray] = None,
    p_off_grid: Optional[np.ndarray] = None,
    p_off_nudge: float = 1.03,
) -> dict:
    """One iteration: plumes → IPW distribution → grid MLE with joint likelihood.

    Key differences from single-tech version:
      - κ_sc computed from IPW sample (NOT κ_s · κ_c).
      - Forward algorithm dispatches on deployment config at each time step.
      - IPW weights = 1/P(plume detected across its span).
    """
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

    alpha = np.ones(n_grid)
    total_alpha = np.zeros(n_grid)
    for t in range(obs.T - 1, -1, -1):
        alpha = p_miss_2d[t] * (p_off_in + (1.0 - p_off_in) * alpha)
        total_alpha += alpha

    p_det_grid = np.clip(1.0 - total_alpha / obs.T, eps, 1.0)

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

    # ── Forward algorithm with correct joint emission probs ──────
    observed_times = obs.observed_times()
    if len(observed_times) < 2:
        return dict(
            p_on_new=p_on_in, p_off_new=p_off_in,
            plumes=plumes, plume_sizes=plume_sizes.tolist(),
            ipw_weights=weights.tolist(), weighted_mean=mu_when_emitting,
            kappa_values=kappa_values, kappa_sc=kappa_sc,
            log_likelihood=-np.inf,
        )

    f_s = tech_specs[0].false_positive_rate
    f_c = tech_specs[1].false_positive_rate

    # ── Grid search over (p_on, p_off) ───────────────────────────
    deltas = np.diff(observed_times).astype(float)

    if p_on_grid is None:
        p_on_grid = np.arange(0.0125, 0.2, 0.0125)
    if p_off_grid is None:
        p_off_grid = np.arange(0.0125, 0.2, 0.0125)

    eff_lb = min(transition_bounds[0],
                 float(p_on_grid.min()), float(p_off_grid.min()))
    eff_ub = transition_bounds[1]

    emit_on = np.ones(len(observed_times))
    emit_off = np.ones(len(observed_times))

    for j, t in enumerate(observed_times):
        has_snap = bool(obs.snap_mask[t])
        has_cont = bool(obs.cont_mask[t])
        d_s = bool(obs.snap_detected[t]) if has_snap else False
        d_c = bool(obs.cont_detected[t]) if has_cont else False

        if has_snap and has_cont:
            if d_s and d_c:
                emit_on[j] = kappa_sc
                emit_off[j] = f_s * f_c
            elif d_s and not d_c:
                emit_on[j] = kappa_s - kappa_sc
                emit_off[j] = f_s * (1 - f_c)
            elif not d_s and d_c:
                emit_on[j] = kappa_c - kappa_sc
                emit_off[j] = (1 - f_s) * f_c
            else:
                emit_on[j] = 1 - kappa_s - kappa_c + kappa_sc
                emit_off[j] = (1 - f_s) * (1 - f_c)
        elif has_snap:
            if d_s:
                emit_on[j] = kappa_s
                emit_off[j] = f_s
            else:
                emit_on[j] = 1 - kappa_s
                emit_off[j] = 1 - f_s
        elif has_cont:
            if d_c:
                emit_on[j] = kappa_c
                emit_off[j] = f_c
            else:
                emit_on[j] = 1 - kappa_c
                emit_off[j] = 1 - f_c

        emit_on[j] = max(emit_on[j], eps)
        emit_off[j] = max(emit_off[j], eps)

    def _ll(p_on_t: float, p_off_t: float) -> float:
        if not (eff_lb <= p_on_t <= eff_ub):
            return -np.inf
        if not (eff_lb <= p_off_t <= eff_ub):
            return -np.inf
        denom = p_on_t + p_off_t
        if denom < eps:
            return -np.inf
        pi0 = p_off_t / denom
        pi1 = p_on_t / denom
        r = 1.0 - p_on_t - p_off_t

        alpha_0 = pi0 * emit_off[0]
        alpha_1 = pi1 * emit_on[0]
        s = alpha_0 + alpha_1
        if s <= eps:
            return -np.inf
        alpha_0 /= s
        alpha_1 /= s
        ll = np.log(s)

        for k in range(len(deltas)):
            rn = r ** deltas[k]
            t00 = pi0 + pi1 * rn
            t01 = pi1 - pi1 * rn
            t10 = pi0 - pi0 * rn
            t11 = pi1 + pi0 * rn

            pred_0 = alpha_0 * t00 + alpha_1 * t10
            pred_1 = alpha_0 * t01 + alpha_1 * t11

            a0 = emit_off[k + 1] * pred_0
            a1 = emit_on[k + 1] * pred_1
            s = a0 + a1
            if s <= eps:
                return -np.inf
            alpha_0 = a0 / s
            alpha_1 = a1 / s
            ll += np.log(s)
        return float(ll)

    best_ll = -np.inf
    best_p_on = p_on_in
    best_p_off = p_off_in
    for p_on_c in p_on_grid:
        for p_off_c in p_off_grid:
            ll = _ll(p_on_c, p_off_c * p_off_nudge)
            if ll > best_ll:
                best_ll = ll
                best_p_on = p_on_c
                best_p_off = p_off_c

    return dict(
        p_on_new=best_p_on, p_off_new=best_p_off,
        plumes=plumes, plume_sizes=plume_sizes.tolist(),
        ipw_weights=weights.tolist(), weighted_mean=mu_when_emitting,
        kappa_values=kappa_values, kappa_sc=kappa_sc,
        log_likelihood=best_ll,
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
    max_iter: int = 20,
    tol: float = 1e-4,
    mean_tol: Optional[float] = None,
    transition_bounds: tuple = (0.01, 0.95),
    eps: float = 1e-10,
    noise_override: Optional[dict] = None,
    decision_threshold: float = 0.5,
    p_on_grid: Optional[np.ndarray] = None,
    p_off_grid: Optional[np.ndarray] = None,
    p_off_nudge: float = 1.03,
) -> dict:
    """
    Run the self-consistent loop estimator on dual-channel observations.

    Returns dict with keys: mean, p_on, p_off, mu_emit,
    empirical_distribution, converged, n_iterations.
    """
    p_on, p_off = init_p_on, init_p_off
    converged = False
    n_iterations = 0
    prev_mean = None
    mean_tol_val = mean_tol if mean_tol is not None else tol
    plume_sizes = np.array([])
    mu_emit = 0.0

    for iteration in range(max_iter):
        step = _run_one_step(
            obs, tech_specs, p_on, p_off,
            max_gap=max_gap, decision_threshold=decision_threshold,
            transition_bounds=transition_bounds, eps=eps,
            noise_override=noise_override,
            p_on_grid=p_on_grid, p_off_grid=p_off_grid,
            p_off_nudge=p_off_nudge,
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
            n_iterations = iteration
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
        "converged": converged,
        "n_iterations": n_iterations,
    }
