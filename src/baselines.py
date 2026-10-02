"""
Baseline (non-MLE) estimators: naive, POD-weighted, MLE-ungrouped,
and MLE-ungrouped-unbiased.

Adapted for the dual-channel SiteObservations structure.
Each technology's observations are counted separately.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .utils import SiteObservations, TechnologySpec


# ═══════════════════════════════════════════════════════════════════
# Noise-corrected IPW helpers for logistic POD q(e) = 1/(1+(θ/e)^k)
#
# For M = e·exp(ε) with ε ~ N(μ_ε, σ_ε²):
#   E[(θ/M)^k] = (θ/e)^k · exp(−k·μ_ε + k²·σ_ε²/2)
#   E[M^(1−k)] = e^(1−k) · exp((1−k)·μ_ε + (1−k)²·σ_ε²/2)
#
# Standard single-sensor noise: μ_ε = −σ²/2, σ_ε = σ
# Combined (inv-var) noise:     μ_ε = −σ_c², σ_ε = σ_c
#   where σ_c² = σ_s²·σ_c² / (σ_s² + σ_c²)
# ═══════════════════════════════════════════════════════════════════

def _f_kappa(m, theta, k, noise_mu, noise_sigma):
    """Corrected 1/q(e): E[f(M)] = 1/q(e) exactly."""
    corr = np.exp(k * noise_mu - k**2 * noise_sigma**2 / 2)
    return 1.0 + (theta / np.maximum(m, 1e-12))**k * corr


def _f_mu(m, theta, k, noise_mu, noise_sigma):
    """Corrected e/q(e): E[f(M)] = e/q(e) exactly."""
    C1 = np.exp(-noise_mu - noise_sigma**2 / 2)
    C2 = np.exp(-(1 - k) * noise_mu - (1 - k)**2 * noise_sigma**2 / 2)
    return C1 * m + C2 * theta**k * np.maximum(m, 1e-12)**(1 - k)


def naive_baseline(obs: SiteObservations) -> float:
    """
    Naive (LUE): mu_hat = sum(detected measurements) / n_observations.

    Unbiased when POD = 1; downward-biased otherwise because
    non-detections are treated as zeros.
    """
    n_obs = obs.n_total_observations()
    if n_obs == 0:
        return 0.0

    total = 0.0
    snap_det = obs.snap_mask & obs.snap_detected
    cont_det = obs.cont_mask & obs.cont_detected
    total += float(np.nansum(obs.snap_measurements[snap_det]))
    total += float(np.nansum(obs.cont_measurements[cont_det]))
    return total / n_obs


def pod_weighted_baseline(
    obs: SiteObservations,
    tech_specs: Sequence[TechnologySpec],
    eps: float = 1e-6,
) -> float:
    """
    Horvitz-Thompson IPW: mu_hat = sum(measurement / POD) / n_observations.

    Corrects for detection probability; still ignores temporal correlation.
    """
    n_obs = obs.n_total_observations()
    if n_obs == 0:
        return 0.0

    total = 0.0

    # Snapshot detections
    for t in np.flatnonzero(obs.snap_mask & obs.snap_detected):
        m = obs.snap_measurements[t]
        pod = tech_specs[0].pod.probability(np.array([m]))[0]
        total += m / max(pod, eps)

    # Continuous detections
    for t in np.flatnonzero(obs.cont_mask & obs.cont_detected):
        m = obs.cont_measurements[t]
        pod = tech_specs[1].pod.probability(np.array([m]))[0]
        total += m / max(pod, eps)

    return total / n_obs


# ═══════════════════════════════════════════════════════════════════
# MLE-ungrouped: per-detection IPW + forward algorithm (no loop)
# ═══════════════════════════════════════════════════════════════════

def _sensor_sigma(spec: TechnologySpec) -> float:
    if spec.sensor is not None and hasattr(spec.sensor, "sigma"):
        return max(float(spec.sensor.sigma), 1e-6)
    return 1e-6


def mle_simple(
    obs: SiteObservations,
    tech_specs: Sequence[TechnologySpec],
    transition_bounds: tuple = (0.01, 0.95),
    eps: float = 1e-10,
    p_on_grid: Optional[np.ndarray] = None,
    p_off_grid: Optional[np.ndarray] = None,
    p_off_nudge: float = 1.03,
) -> dict:
    """One-shot MLE: per-detection IPW for mu_emit/kappas, then forward algorithm.

    Steps:
      1. Per-detection IPW (Hajek estimator) to get mu_emit, kappa_s,
         kappa_c, kappa_sc.  Simultaneous detections are combined into
         a single record via inverse-variance weighting in log-space.
      2. Forward algorithm + grid search over (p_on, p_off) using the
         kappas from step 1.
      3. mu_hat = pi_on_hat * mu_emit_hat.

    Returns dict with keys: mean, p_on, p_off, mu_emit.
    """
    sigma_s = _sensor_sigma(tech_specs[0])
    sigma_c = _sensor_sigma(tech_specs[1])
    inv_var_s = 1.0 / sigma_s ** 2
    inv_var_c = 1.0 / sigma_c ** 2

    # ── Step 1: per-detection IPW ─────────────────────────────────
    det_E: list[float] = []
    det_w: list[float] = []
    det_qs: list[float] = []
    det_qc: list[float] = []

    for t in range(obs.T):
        has_snap = bool(obs.snap_mask[t])
        has_cont = bool(obs.cont_mask[t])
        det_s = has_snap and bool(obs.snap_detected[t])
        det_c = has_cont and bool(obs.cont_detected[t])
        if not det_s and not det_c:
            continue

        both_deployed = has_snap and has_cont

        if det_s and det_c:
            log_comb = (np.log(max(obs.snap_measurements[t], 1e-12)) * inv_var_s
                        + np.log(max(obs.cont_measurements[t], 1e-12)) * inv_var_c
                        ) / (inv_var_s + inv_var_c)
            E = float(np.exp(log_comb))
            qs = float(tech_specs[0].pod.probability(np.array([E]))[0])
            qc = float(tech_specs[1].pod.probability(np.array([E]))[0])
            q_det = 1.0 - (1.0 - qs) * (1.0 - qc)
        elif det_s:
            E = float(obs.snap_measurements[t])
            qs = float(tech_specs[0].pod.probability(np.array([E]))[0])
            qc = float(tech_specs[1].pod.probability(np.array([E]))[0])
            q_det = (1.0 - (1.0 - qs) * (1.0 - qc)) if both_deployed else qs
        else:
            E = float(obs.cont_measurements[t])
            qs = float(tech_specs[0].pod.probability(np.array([E]))[0])
            qc = float(tech_specs[1].pod.probability(np.array([E]))[0])
            q_det = (1.0 - (1.0 - qs) * (1.0 - qc)) if both_deployed else qc

        det_E.append(E)
        det_w.append(1.0 / max(q_det, eps))
        det_qs.append(qs)
        det_qc.append(qc)

    if len(det_E) < 2:
        return dict(mean=np.nan, p_on=np.nan, p_off=np.nan, mu_emit=np.nan)

    sizes = np.array(det_E)
    weights = np.array(det_w)
    qs_arr = np.array(det_qs)
    qc_arr = np.array(det_qc)

    w_sum = float(np.sum(weights))
    mu_emit = float(np.sum(weights * sizes) / w_sum)
    kappa_s = float(np.clip(np.sum(weights * qs_arr) / w_sum, eps, 1 - eps))
    kappa_c = float(np.clip(np.sum(weights * qc_arr) / w_sum, eps, 1 - eps))
    kappa_sc = float(np.clip(
        np.sum(weights * qs_arr * qc_arr) / w_sum, eps, 1 - eps))
    kappa_sc = min(kappa_sc, kappa_s, kappa_c)

    # ── Step 2: forward algorithm + grid search ───────────────────
    observed_times = obs.observed_times()
    if len(observed_times) < 2:
        return dict(mean=np.nan, p_on=np.nan, p_off=np.nan, mu_emit=mu_emit)

    n_obs_t = len(observed_times)
    emit_on = np.ones(n_obs_t)
    emit_off = np.ones(n_obs_t)

    f_s = tech_specs[0].false_positive_rate
    f_c = tech_specs[1].false_positive_rate

    for j, t in enumerate(observed_times):
        has_snap = bool(obs.snap_mask[t])
        has_cont = bool(obs.cont_mask[t])
        det_s = bool(obs.snap_detected[t]) if has_snap else False
        det_c = bool(obs.cont_detected[t]) if has_cont else False

        if has_snap and has_cont:
            if det_s and det_c:
                emit_on[j] = kappa_sc
                emit_off[j] = f_s * f_c
            elif det_s:
                emit_on[j] = kappa_s - kappa_sc
                emit_off[j] = f_s * (1 - f_c)
            elif det_c:
                emit_on[j] = kappa_c - kappa_sc
                emit_off[j] = (1 - f_s) * f_c
            else:
                emit_on[j] = 1 - kappa_s - kappa_c + kappa_sc
                emit_off[j] = (1 - f_s) * (1 - f_c)
        elif has_snap:
            if det_s:
                emit_on[j] = kappa_s
                emit_off[j] = f_s
            else:
                emit_on[j] = 1 - kappa_s
                emit_off[j] = 1 - f_s
        elif has_cont:
            if det_c:
                emit_on[j] = kappa_c
                emit_off[j] = f_c
            else:
                emit_on[j] = 1 - kappa_c
                emit_off[j] = 1 - f_c

        emit_on[j] = max(emit_on[j], eps)
        emit_off[j] = max(emit_off[j], eps)

    deltas = np.diff(observed_times).astype(float)

    best_ll = -np.inf
    best_p_on = 0.025
    best_p_off = 0.10
    _p_on_g = p_on_grid if p_on_grid is not None else np.arange(0.0125, 0.2, 0.0125)
    _p_off_g = p_off_grid if p_off_grid is not None else np.arange(0.0125, 0.2, 0.0125)
    _eff_lb = min(transition_bounds[0],
                  float(_p_on_g.min()), float(_p_off_g.min()))

    for p_on_c in _p_on_g:
        for p_off_c in _p_off_g:
            p_off_c_eff = p_off_c * p_off_nudge
            if not (_eff_lb <= p_on_c <= transition_bounds[1]):
                continue
            if not (_eff_lb <= p_off_c_eff <= transition_bounds[1]):
                continue
            denom = p_on_c + p_off_c_eff
            if denom < eps:
                continue
            pi0 = p_off_c_eff / denom
            pi1 = p_on_c / denom
            r = 1.0 - p_on_c - p_off_c_eff

            a0 = pi0 * emit_off[0]
            a1 = pi1 * emit_on[0]
            s = a0 + a1
            if s <= eps:
                continue
            a0 /= s
            a1 /= s
            ll = np.log(s)

            bad = False
            for k in range(len(deltas)):
                rn = r ** deltas[k]
                pred_0 = a0 * (pi0 + pi1 * rn) + a1 * (pi0 - pi0 * rn)
                pred_1 = a0 * (pi1 - pi1 * rn) + a1 * (pi1 + pi0 * rn)
                a0 = emit_off[k + 1] * pred_0
                a1 = emit_on[k + 1] * pred_1
                s = a0 + a1
                if s <= eps:
                    bad = True
                    break
                a0 /= s
                a1 /= s
                ll += np.log(s)

            if bad:
                continue
            if ll > best_ll:
                best_ll = ll
                best_p_on = p_on_c
                best_p_off = p_off_c

    # ── Step 3: combine ───────────────────────────────────────────
    pi_on = best_p_on / (best_p_on + best_p_off)
    mu_hat = pi_on * mu_emit

    return dict(mean=mu_hat, p_on=best_p_on, p_off=best_p_off,
                mu_emit=mu_emit)


# ═══════════════════════════════════════════════════════════════════
# MLE-ungrouped-unbiased: noise-corrected IPW + forward algorithm
# ═══════════════════════════════════════════════════════════════════

def mle_simple_unbiased(
    obs: SiteObservations,
    tech_specs: Sequence[TechnologySpec],
    transition_bounds: tuple = (0.01, 0.95),
    eps: float = 1e-10,
    p_on_grid: Optional[np.ndarray] = None,
    p_off_grid: Optional[np.ndarray] = None,
) -> dict:
    """MLE-ungrouped with Jensen-corrected IPW weights.

    Uses analytical correction factors for the logistic POD so that
    E[f_kappa(M)] = 1/q(e) and E[f_mu(M)] = e/q(e) exactly,
    eliminating the bias from evaluating POD at noisy measurements.

    Returns dict with keys: mean, p_on, p_off, mu_emit.
    """
    sigma_s = _sensor_sigma(tech_specs[0])
    sigma_c = _sensor_sigma(tech_specs[1])

    theta_s = tech_specs[0].pod.threshold
    k_s = tech_specs[0].pod.slope
    theta_c = tech_specs[1].pod.threshold
    k_c = tech_specs[1].pod.slope

    inv_var_s = 1.0 / sigma_s**2
    inv_var_c = 1.0 / sigma_c**2
    sigma_comb_sq = sigma_s**2 * sigma_c**2 / (sigma_s**2 + sigma_c**2)
    sigma_comb = np.sqrt(sigma_comb_sq)

    # Single-sensor noise: eps ~ N(-sig^2/2, sig^2)
    mu_eps_s = -sigma_s**2 / 2
    mu_eps_c = -sigma_c**2 / 2
    # Combined noise: eps ~ N(-sig_comb^2, sig_comb^2)
    mu_eps_comb = -sigma_comb_sq

    # ── Step 1: noise-corrected IPW ─────────────────────────────────
    sum_fk_s = 0.0        # sum of f_kappa for snap POD
    sum_fk_c = 0.0        # sum of f_kappa for cont POD
    sum_fk_sc = 0.0       # sum of f_kappa for joint q_s*q_c
    sum_fmu_det = 0.0     # sum of f_mu using detecting sensor's POD
    sum_fk_det = 0.0      # sum of f_kappa for detection probability
    N = 0

    for t in range(obs.T):
        has_snap = bool(obs.snap_mask[t])
        has_cont = bool(obs.cont_mask[t])
        det_s = has_snap and bool(obs.snap_detected[t])
        det_c = has_cont and bool(obs.cont_detected[t])
        if not det_s and not det_c:
            continue
        N += 1

        both_deployed = has_snap and has_cont

        if det_s and det_c:
            # Combined measurement, combined noise
            log_comb = (np.log(max(obs.snap_measurements[t], 1e-12)) * inv_var_s
                        + np.log(max(obs.cont_measurements[t], 1e-12)) * inv_var_c
                        ) / (inv_var_s + inv_var_c)
            m = float(np.exp(log_comb))
            mu_n, sig_n = mu_eps_comb, sigma_comb

            fk_s = _f_kappa(m, theta_s, k_s, mu_n, sig_n)
            fk_c = _f_kappa(m, theta_c, k_c, mu_n, sig_n)
            tq_s = 1.0 / max(fk_s, eps)
            tq_c = 1.0 / max(fk_c, eps)
            tq_comb = 1.0 - (1.0 - tq_s) * (1.0 - tq_c)
            fk_det = 1.0 / max(tq_comb, eps)

            # f_mu uses the combined sensor with higher POD (lower threshold)
            # to cancel the detection weight; use ratio approach:
            # mu_emit = sum(f_mu) / sum(f_kappa_det)
            fm = _f_mu(m, theta_s, k_s, mu_n, sig_n)
            fm_c = _f_mu(m, theta_c, k_c, mu_n, sig_n)
            # Use the weighted combination that estimates e/q_comb:
            # e/q_comb = e / (q_s + q_c - q_s*q_c)
            # Approximate via: (tq_s * fm_s_val + tq_c * fm_c_val) / (tq_s + tq_c)
            # where fm gives e/q_j, so tq_j * fm = tq_j * e/q_j ≈ e (for each j)
            # Simpler: use e/q_comb = e * f_kappa_det, but we need E[m * f_kappa_det]
            # Cleanest ratio estimator: f_mu_det = m * C1 * f_kappa_det
            C1 = np.exp(-mu_n - sig_n**2 / 2)
            fm_det = C1 * m * fk_det

        elif det_s:
            m = float(obs.snap_measurements[t])
            mu_n, sig_n = mu_eps_s, sigma_s

            fk_s = _f_kappa(m, theta_s, k_s, mu_n, sig_n)
            fk_c = _f_kappa(m, theta_c, k_c, mu_n, sig_n)

            if both_deployed:
                tq_s = 1.0 / max(fk_s, eps)
                tq_c = 1.0 / max(fk_c, eps)
                tq_comb = 1.0 - (1.0 - tq_s) * (1.0 - tq_c)
                fk_det = 1.0 / max(tq_comb, eps)
            else:
                fk_det = fk_s

            C1 = np.exp(-mu_n - sig_n**2 / 2)
            fm_det = C1 * m * fk_det

        else:  # det_c only
            m = float(obs.cont_measurements[t])
            mu_n, sig_n = mu_eps_c, sigma_c

            fk_s = _f_kappa(m, theta_s, k_s, mu_n, sig_n)
            fk_c = _f_kappa(m, theta_c, k_c, mu_n, sig_n)

            if both_deployed:
                tq_s = 1.0 / max(fk_s, eps)
                tq_c = 1.0 / max(fk_c, eps)
                tq_comb = 1.0 - (1.0 - tq_s) * (1.0 - tq_c)
                fk_det = 1.0 / max(tq_comb, eps)
            else:
                fk_det = fk_c

            C1 = np.exp(-mu_n - sig_n**2 / 2)
            fm_det = C1 * m * fk_det

        sum_fk_s += fk_s
        sum_fk_c += fk_c
        tq_s_val = 1.0 / max(fk_s, eps)
        tq_c_val = 1.0 / max(fk_c, eps)
        sum_fk_sc += 1.0 / max(tq_s_val * tq_c_val, eps)
        sum_fk_det += fk_det
        sum_fmu_det += fm_det

    if N < 2:
        return dict(mean=np.nan, p_on=np.nan, p_off=np.nan, mu_emit=np.nan)

    kappa_s = float(np.clip(N / sum_fk_s, eps, 1 - eps))
    kappa_c = float(np.clip(N / sum_fk_c, eps, 1 - eps))
    kappa_sc = float(np.clip(N / sum_fk_sc, eps, 1 - eps))
    kappa_sc = min(kappa_sc, kappa_s, kappa_c)
    mu_emit = float(sum_fmu_det / sum_fk_det)

    # ── Step 2: forward algorithm + grid search ───────────────────
    # (identical to mle_simple)
    observed_times = obs.observed_times()
    if len(observed_times) < 2:
        return dict(mean=np.nan, p_on=np.nan, p_off=np.nan, mu_emit=mu_emit)

    n_obs_t = len(observed_times)
    emit_on = np.ones(n_obs_t)
    emit_off = np.ones(n_obs_t)

    f_s = tech_specs[0].false_positive_rate
    f_c = tech_specs[1].false_positive_rate

    for j, t in enumerate(observed_times):
        has_snap = bool(obs.snap_mask[t])
        has_cont = bool(obs.cont_mask[t])
        det_s_t = bool(obs.snap_detected[t]) if has_snap else False
        det_c_t = bool(obs.cont_detected[t]) if has_cont else False

        if has_snap and has_cont:
            if det_s_t and det_c_t:
                emit_on[j] = kappa_sc
                emit_off[j] = f_s * f_c
            elif det_s_t:
                emit_on[j] = kappa_s - kappa_sc
                emit_off[j] = f_s * (1 - f_c)
            elif det_c_t:
                emit_on[j] = kappa_c - kappa_sc
                emit_off[j] = (1 - f_s) * f_c
            else:
                emit_on[j] = 1 - kappa_s - kappa_c + kappa_sc
                emit_off[j] = (1 - f_s) * (1 - f_c)
        elif has_snap:
            if det_s_t:
                emit_on[j] = kappa_s
                emit_off[j] = f_s
            else:
                emit_on[j] = 1 - kappa_s
                emit_off[j] = 1 - f_s
        elif has_cont:
            if det_c_t:
                emit_on[j] = kappa_c
                emit_off[j] = f_c
            else:
                emit_on[j] = 1 - kappa_c
                emit_off[j] = 1 - f_c

        emit_on[j] = max(emit_on[j], eps)
        emit_off[j] = max(emit_off[j], eps)

    deltas = np.diff(observed_times).astype(float)

    best_ll = -np.inf
    best_p_on = 0.025
    best_p_off = 0.10
    _p_on_g = p_on_grid if p_on_grid is not None else np.arange(0.0125, 0.2, 0.0125)
    _p_off_g = p_off_grid if p_off_grid is not None else np.arange(0.0125, 0.2, 0.0125)
    _eff_lb = min(transition_bounds[0],
                  float(_p_on_g.min()), float(_p_off_g.min()))

    for p_on_c in _p_on_g:
        for p_off_c in _p_off_g:
            if not (_eff_lb <= p_on_c <= transition_bounds[1]):
                continue
            if not (_eff_lb <= p_off_c <= transition_bounds[1]):
                continue
            denom = p_on_c + p_off_c
            if denom < eps:
                continue
            pi0 = p_off_c / denom
            pi1 = p_on_c / denom
            r = 1.0 - p_on_c - p_off_c

            a0 = pi0 * emit_off[0]
            a1 = pi1 * emit_on[0]
            s = a0 + a1
            if s <= eps:
                continue
            a0 /= s
            a1 /= s
            ll = np.log(s)

            bad = False
            for k in range(len(deltas)):
                rn = r ** deltas[k]
                pred_0 = a0 * (pi0 + pi1 * rn) + a1 * (pi0 - pi0 * rn)
                pred_1 = a0 * (pi1 - pi1 * rn) + a1 * (pi1 + pi0 * rn)
                a0 = emit_off[k + 1] * pred_0
                a1 = emit_on[k + 1] * pred_1
                s = a0 + a1
                if s <= eps:
                    bad = True
                    break
                a0 /= s
                a1 /= s
                ll += np.log(s)

            if bad:
                continue
            if ll > best_ll:
                best_ll = ll
                best_p_on = p_on_c
                best_p_off = p_off_c

    # ── Step 3: combine ───────────────────────────────────────────
    pi_on = best_p_on / (best_p_on + best_p_off)
    mu_hat = pi_on * mu_emit

    return dict(mean=mu_hat, p_on=best_p_on, p_off=best_p_off,
                mu_emit=mu_emit)
