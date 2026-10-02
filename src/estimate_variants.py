"""
Variant MLE estimators that incorporate plume-linking information
into the forward algorithm for improved p_on / p_off estimation.

Three variants (all leave mu_emit / kappa / IPW steps unchanged):

  1. MLE-constrained:   Within-plume observed time steps forced to ON.
  2. MLE-plume-kappa:   Plume-specific kappa at within-plume time steps.
  3. MLE-penalized:     Geometric(p_off) duration penalty in grid search.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .utils import SiteObservations, TechnologySpec
from .estimate import identify_plumes, _combined_log_estimate, Plume


def _run_one_step_variant(
    obs: SiteObservations,
    tech_specs: Sequence[TechnologySpec],
    p_on_in: float,
    p_off_in: float,
    mode: str,
    max_gap: int = 3,
    decision_threshold: float = 0.5,
    transition_bounds: tuple = (0.01, 0.95),
    eps: float = 1e-10,
    noise_override: Optional[dict] = None,
    p_on_grid: Optional[np.ndarray] = None,
    p_off_grid: Optional[np.ndarray] = None,
) -> dict:
    """One iteration with plume-aware forward algorithm.

    *mode* selects the variant:
      'constrained'  – force within-plume time steps to ON
      'plume_kappa'  – use plume-specific kappa at within-plume steps
      'penalized'    – add Geometric duration penalty to log-likelihood
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

    # ── Plume sizes (lognormal-bias-corrected, same as original) ──
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

    # ── Backward recursion for detection probabilities (unchanged) ──
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
    for m in range(len(plumes)):
        log_e = np.log(max(plume_sizes[m], 0.5))
        log_e = np.clip(log_e, log_e_lo, log_e_hi)
        idx = (log_e - log_e_lo) / (log_e_hi - log_e_lo) * (n_grid - 1)
        i0 = int(idx)
        i1 = min(i0 + 1, n_grid - 1)
        frac = idx - i0
        plume_det_probs[m] = max(
            (1.0 - frac) * p_det_grid[i0] + frac * p_det_grid[i1], eps,
        )

    # ── IPW weights & mu_emit ────────────────────────────────────
    weights = 1.0 / plume_det_probs
    mu_when_emitting = float(np.sum(weights * plume_sizes) / np.sum(weights))

    # ── Aggregate kappa (used for time steps outside plume spans) ─
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

    # ── Forward-algorithm setup ──────────────────────────────────
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

    # ── Map observed times to plume spans ─────────────────────────
    plume_at_time: dict[int, int] = {}
    for m, p in enumerate(plumes):
        for t in range(p.indices[0], p.indices[-1] + 1):
            plume_at_time[t] = m

    # ── Plume-specific kappas (for mode='plume_kappa') ────────────
    plume_ks = tech_specs[0].pod.probability(plume_sizes)
    plume_kc = tech_specs[1].pod.probability(plume_sizes)
    plume_ksc = plume_ks * plume_kc  # independent given fixed size

    # ── Build emission probability arrays ────────────────────────
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

        # Select kappa: plume-specific or aggregate
        if mode == 'plume_kappa' and t in plume_at_time:
            m = plume_at_time[t]
            ks = float(plume_ks[m])
            kc = float(plume_kc[m])
            ksc_val = float(plume_ksc[m])
        else:
            ks = kappa_s
            kc = kappa_c
            ksc_val = kappa_sc

        if has_snap and has_cont:
            if d_s and d_c:
                emit_on[j] = ksc_val
                emit_off[j] = f_s * f_c
            elif d_s and not d_c:
                emit_on[j] = ks - ksc_val
                emit_off[j] = f_s * (1 - f_c)
            elif not d_s and d_c:
                emit_on[j] = kc - ksc_val
                emit_off[j] = (1 - f_s) * f_c
            else:
                emit_on[j] = 1 - ks - kc + ksc_val
                emit_off[j] = (1 - f_s) * (1 - f_c)
        elif has_snap:
            if d_s:
                emit_on[j] = ks
                emit_off[j] = f_s
            else:
                emit_on[j] = 1 - ks
                emit_off[j] = 1 - f_s
        elif has_cont:
            if d_c:
                emit_on[j] = kc
                emit_off[j] = f_c
            else:
                emit_on[j] = 1 - kc
                emit_off[j] = 1 - f_c

        emit_on[j] = max(emit_on[j], eps)
        emit_off[j] = max(emit_off[j], eps)

    # ── Mode-specific: constrain within-plume steps to ON ────────
    if mode == 'constrained':
        for j, t in enumerate(observed_times):
            if t in plume_at_time:
                emit_off[j] = eps

    # ── Plume durations for penalized mode ───────────────────────
    plume_durations = np.array([p.duration for p in plumes])

    # ── Grid search ──────────────────────────────────────────────
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

        if mode == 'penalized':
            for D in plume_durations:
                ll += (D - 1) * np.log(max(1 - p_off_t, eps))
                ll += np.log(max(p_off_t, eps))

        return float(ll)

    best_ll = -np.inf
    best_p_on = p_on_in
    best_p_off = p_off_in
    for p_on_c in p_on_grid:
        for p_off_c in p_off_grid:
            ll = _ll(p_on_c, p_off_c)
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
# Loop wrappers
# ═══════════════════════════════════════════════════════════════════

def _loop_variant(
    obs: SiteObservations,
    tech_specs: Sequence[TechnologySpec],
    T: int,
    mode: str,
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
) -> dict:
    p_on, p_off = init_p_on, init_p_off
    converged = False
    n_iterations = 0
    prev_mean = None
    mean_tol_val = mean_tol if mean_tol is not None else tol
    plume_sizes = np.array([])
    mu_emit = 0.0

    for iteration in range(max_iter):
        step = _run_one_step_variant(
            obs, tech_specs, p_on, p_off, mode=mode,
            max_gap=max_gap, decision_threshold=decision_threshold,
            transition_bounds=transition_bounds, eps=eps,
            noise_override=noise_override,
            p_on_grid=p_on_grid, p_off_grid=p_off_grid,
        )

        p_on_new = step["p_on_new"]
        p_off_new = step["p_off_new"]
        mu_emit = step["weighted_mean"]
        plume_sizes = np.array(step["plume_sizes"])

        pi_emit = p_on_new / (p_on_new + p_off_new)
        current_mean = mu_emit * pi_emit

        params_ok = (abs(p_on_new - p_on) < tol
                     and abs(p_off_new - p_off) < tol)
        mean_ok = (prev_mean is not None
                   and abs(current_mean - prev_mean) < mean_tol_val)
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


def loop_mle_constrained(obs, tech_specs, T, **kwargs):
    """MLE with within-plume time steps forced to ON in forward alg."""
    return _loop_variant(obs, tech_specs, T, mode='constrained', **kwargs)


def loop_mle_plume_kappa(obs, tech_specs, T, **kwargs):
    """MLE with plume-specific detection probability in forward alg."""
    return _loop_variant(obs, tech_specs, T, mode='plume_kappa', **kwargs)


def loop_mle_penalized(obs, tech_specs, T, **kwargs):
    """MLE with Geometric(p_off) duration penalty in grid search."""
    return _loop_variant(obs, tech_specs, T, mode='penalized', **kwargs)
