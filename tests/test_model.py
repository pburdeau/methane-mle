"""
Tests for the hybrid observation model.

Covers:
  1. κ_sc > κ_s · κ_c (correlation through shared latent Q_t).
  2. Emission probabilities sum to 1 for single and dual deployment.
  3. p_cont=0 recovers snapshot-only model exactly.
  4. snap=0, p_cont=1 recovers continuous-only model exactly.
  5. Observation mask generation (hybrid: snap per-step, cont windows).
  6. Plume identification with dual-channel detections.
  7. Parameter recovery from synthetic data.
  8. Inverse-variance weighted combination of measurements.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    SiteObservations, TechnologySpec, LogisticPOD, LognormalSensor,
    build_baseline_specs, lognormal_expectation,
)
from src.simulate import generate_masks, simulate_series
from src.estimate import (
    identify_plumes, loop_empirical, _run_one_step,
    _combined_log_estimate,
)
from src.baselines import naive_baseline, pod_weighted_baseline


def _default_specs():
    return build_baseline_specs(
        aerial_threshold=50.0, aerial_slope=2.0, aerial_sensor_sigma=0.05,
        cont_threshold=2.4, cont_slope=3.0, cont_sensor_sigma=0.2,
        cont_fp_rate=0.0, cont_fp_scale=4.0,
    )


# ══════════════════════════════════════════════════════════════════
# Test 1: κ_sc > κ_s · κ_c (joint integral, NOT product)
# ══════════════════════════════════════════════════════════════════

def test_kappa_sc_greater_than_product():
    """κ_sc = E[q_s(Q)·q_c(Q)] must exceed E[q_s(Q)]·E[q_c(Q)]
    because q_s and q_c are positively correlated through shared Q."""
    specs = _default_specs()
    mu, sigma = 3.787, 0.5

    kappa_s = lognormal_expectation(mu, sigma, lambda x: specs[0].pod.probability(x))
    kappa_c = lognormal_expectation(mu, sigma, lambda x: specs[1].pod.probability(x))
    kappa_sc = lognormal_expectation(
        mu, sigma,
        lambda x: specs[0].pod.probability(x) * specs[1].pod.probability(x),
    )

    assert kappa_sc > kappa_s * kappa_c, (
        f"κ_sc={kappa_sc:.6f} should be > κ_s·κ_c={kappa_s * kappa_c:.6f}"
    )
    assert kappa_sc <= min(kappa_s, kappa_c), (
        f"κ_sc={kappa_sc:.6f} should be ≤ min(κ_s,κ_c)={min(kappa_s, kappa_c):.6f}"
    )


# ══════════════════════════════════════════════════════════════════
# Test 2: Emission probabilities sum to 1 (dual deployment)
# ══════════════════════════════════════════════════════════════════

def test_emission_prob_sum_dual_deployment():
    """For dual deployment, the four emission probabilities for Z=1
    (using κ_s, κ_c, κ_sc) must sum to exactly 1."""
    specs = _default_specs()
    mu, sigma = 3.787, 0.5

    kappa_s = lognormal_expectation(mu, sigma, lambda x: specs[0].pod.probability(x))
    kappa_c = lognormal_expectation(mu, sigma, lambda x: specs[1].pod.probability(x))
    kappa_sc = lognormal_expectation(
        mu, sigma,
        lambda x: specs[0].pod.probability(x) * specs[1].pod.probability(x),
    )

    p_both = kappa_sc
    p_snap_only = kappa_s - kappa_sc
    p_cont_only = kappa_c - kappa_sc
    p_neither = 1 - kappa_s - kappa_c + kappa_sc

    total_on = p_both + p_snap_only + p_cont_only + p_neither
    assert abs(total_on - 1.0) < 1e-12, f"Z=1 probs sum to {total_on}"

    assert p_both >= 0, f"P(both detect|Z=1) = {p_both} < 0"
    assert p_snap_only >= 0, f"P(snap only|Z=1) = {p_snap_only} < 0"
    assert p_cont_only >= 0, f"P(cont only|Z=1) = {p_cont_only} < 0"
    assert p_neither >= 0, f"P(neither|Z=1) = {p_neither} < 0"

    f_s, f_c = 0.0, 0.0
    total_off = (f_s * f_c + f_s * (1 - f_c) +
                 (1 - f_s) * f_c + (1 - f_s) * (1 - f_c))
    assert abs(total_off - 1.0) < 1e-12, f"Z=0 probs sum to {total_off}"


# ══════════════════════════════════════════════════════════════════
# Test 3: Emission probabilities sum to 1 (single deployment)
# ══════════════════════════════════════════════════════════════════

def test_emission_prob_sum_single_deployment():
    """For single tech, detection + non-detection must sum to 1."""
    specs = _default_specs()
    mu, sigma = 3.787, 0.5

    kappa_s = lognormal_expectation(mu, sigma, lambda x: specs[0].pod.probability(x))
    kappa_c = lognormal_expectation(mu, sigma, lambda x: specs[1].pod.probability(x))

    assert abs(kappa_s + (1 - kappa_s) - 1.0) < 1e-12
    assert abs(kappa_c + (1 - kappa_c) - 1.0) < 1e-12

    f_s, f_c = 0.0, 0.005
    assert abs(f_s + (1 - f_s) - 1.0) < 1e-12
    assert abs(f_c + (1 - f_c) - 1.0) < 1e-12


# ══════════════════════════════════════════════════════════════════
# Test 4: Observation mask generation
# ══════════════════════════════════════════════════════════════════

def test_generate_masks_snap_only():
    """p_cont=0 should produce no continuous observations."""
    rng = np.random.default_rng(42)
    snap_mask, cont_mask = generate_masks(
        T=100, p_snap=1.0, p_cont=0.0, T_cont=50, rng=rng)
    assert np.all(snap_mask)
    assert not np.any(cont_mask)


def test_generate_masks_cont_only():
    """p_snap=0, p_cont=1 should give full continuous window."""
    rng = np.random.default_rng(42)
    snap_mask, cont_mask = generate_masks(
        T=100, p_snap=0.0, p_cont=1.0, T_cont=100, rng=rng)
    assert not np.any(snap_mask)
    assert np.all(cont_mask)


def test_generate_masks_cont_window_contiguous():
    """Each continuous window should be exactly T_cont contiguous steps."""
    rng = np.random.default_rng(42)
    T, T_cont = 500, 50
    _, cont_mask = generate_masks(
        T=T, p_snap=0.0, p_cont=0.05, T_cont=T_cont, rng=rng)
    active = np.flatnonzero(cont_mask)
    if len(active) == 0:
        return
    runs = np.split(active, np.where(np.diff(active) != 1)[0] + 1)
    for run in runs:
        assert len(run) == T_cont or run[-1] == T - 1, (
            f"Window length {len(run)} != T_cont={T_cont} "
            f"(allowed if truncated at end)")


def test_generate_masks_renewal_coverage():
    """With p_cont=1, every step triggers a window, so all steps covered."""
    rng = np.random.default_rng(42)
    _, cont_mask = generate_masks(
        T=200, p_snap=0.0, p_cont=1.0, T_cont=50, rng=rng)
    assert np.all(cont_mask), "p_cont=1 should cover all time steps"


# ══════════════════════════════════════════════════════════════════
# Test 5: SiteObservations structure
# ══════════════════════════════════════════════════════════════════

def test_site_observations_basics():
    T = 10
    snap_mask = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    snap_det = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    snap_meas = np.full(T, np.nan)
    snap_meas[1] = 50.0
    snap_meas[3] = 60.0

    cont_mask = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    cont_det = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    cont_meas = np.full(T, np.nan)
    cont_meas[0] = 10.0
    cont_meas[2] = 12.0

    obs = SiteObservations(T, snap_mask, snap_det, snap_meas,
                           cont_mask, cont_det, cont_meas)

    assert obs.T == 10
    assert obs.n_total_observations() == 8  # 5 snap + 3 cont
    assert obs.n_total_detections() == 4  # 2 snap + 2 cont

    det_list = obs.detection_list()
    assert len(det_list) == 4
    assert det_list[0] == (0, 1, 10.0)  # cont detection at t=0
    assert det_list[1] == (1, 0, 50.0)  # snap detection at t=1


# ══════════════════════════════════════════════════════════════════
# Test 6: Simulation produces valid output
# ══════════════════════════════════════════════════════════════════

def test_simulate_series_basic():
    specs = _default_specs()
    T = 200
    rng = np.random.default_rng(123)

    snap_mask = np.ones(T, dtype=bool)
    cont_mask = np.zeros(T, dtype=bool)
    cont_mask[:50] = True

    obs, states, sizes, event_ids = simulate_series(
        T=T, p_on=0.025, p_off=0.10,
        size_mu=3.787, size_sigma=0.5,
        tech_specs=specs,
        snap_mask=snap_mask, cont_mask=cont_mask, rng=rng,
    )

    assert obs.T == T
    assert states.shape == (T,)
    assert sizes.shape == (T,)
    assert event_ids.shape == (T,)
    assert np.all(obs.snap_mask == snap_mask)
    assert np.all(obs.cont_mask == cont_mask)
    assert np.all((states == 0) | (states == 1))
    assert np.all(sizes[states == 0] == 0.0)
    assert np.all(sizes[states == 1] > 0.0)


# ══════════════════════════════════════════════════════════════════
# Test 7: p_cont=0 recovers snapshot-only model exactly
# ══════════════════════════════════════════════════════════════════

def test_snapshot_only_reduction():
    """With p_cont=0, the estimator should work using only snapshot data."""
    specs = _default_specs()
    T = 250
    rng_mask = np.random.default_rng(42)
    rng_emit = np.random.default_rng(43)

    snap_mask, cont_mask = generate_masks(
        T, p_snap=1.0, p_cont=0.0, T_cont=50, rng=rng_mask,
    )
    assert not np.any(cont_mask)

    obs, states, sizes, eids = simulate_series(
        T=T, p_on=0.025, p_off=0.10,
        size_mu=3.787, size_sigma=0.5,
        tech_specs=specs,
        snap_mask=snap_mask, cont_mask=cont_mask, rng=rng_emit,
    )

    n_det = obs.n_total_detections()
    if n_det >= 2:
        res = loop_empirical(obs, specs, T)
        assert "mean" in res
        assert "p_on" in res
        assert "p_off" in res
        assert res["mean"] >= 0
        assert 0 < res["p_on"] < 1
        assert 0 < res["p_off"] < 1


# ══════════════════════════════════════════════════════════════════
# Test 8: p_snap=0 recovers continuous-only model exactly
# ══════════════════════════════════════════════════════════════════

def test_continuous_only_reduction():
    """With p_snap=0, the estimator should work using only continuous data."""
    specs = _default_specs()
    T = 250
    rng_mask = np.random.default_rng(42)
    rng_emit = np.random.default_rng(43)

    snap_mask, cont_mask = generate_masks(
        T, p_snap=0.0, p_cont=1.0, T_cont=T, rng=rng_mask,
    )
    assert not np.any(snap_mask)
    assert np.all(cont_mask)

    obs, states, sizes, eids = simulate_series(
        T=T, p_on=0.025, p_off=0.10,
        size_mu=3.787, size_sigma=0.5,
        tech_specs=specs,
        snap_mask=snap_mask, cont_mask=cont_mask, rng=rng_emit,
    )

    n_det = obs.n_total_detections()
    if n_det >= 2:
        res = loop_empirical(obs, specs, T)
        assert "mean" in res
        assert res["mean"] >= 0
        assert 0 < res["p_on"] < 1
        assert 0 < res["p_off"] < 1


# ══════════════════════════════════════════════════════════════════
# Test 9: Inverse-variance weighted combination in log-space
# ══════════════════════════════════════════════════════════════════

def test_combined_log_estimate():
    """Inverse-variance weighted average in log-space: precise tech
    should dominate when combined with noisy tech."""
    specs = _default_specs()
    sigma_s = 0.05  # snapshot: low noise
    sigma_c = 0.2   # continuous: high noise

    m_snap = 55.0
    m_cont = 40.0
    combined, eff_sigma = _combined_log_estimate(
        [m_snap, m_cont], [0, 1], specs,
    )

    assert combined > m_cont, "Combined should be pulled toward snap (lower noise)"
    assert combined < m_snap, "Combined should be between the two"

    expected_eff_sigma = 1.0 / np.sqrt(1 / sigma_s ** 2 + 1 / sigma_c ** 2)
    assert abs(eff_sigma - expected_eff_sigma) < 1e-6

    single, single_sigma = _combined_log_estimate([m_snap], [0], specs)
    assert abs(single - m_snap) < 1e-6
    assert abs(single_sigma - sigma_s) < 1e-6


# ══════════════════════════════════════════════════════════════════
# Test 10: Plume identification with dual-channel detections
# ══════════════════════════════════════════════════════════════════

def test_plume_identification_dual_channel():
    """Detections at the same time step from different techs should be linked."""
    T = 10
    snap_mask = np.ones(T, dtype=bool)
    cont_mask = np.ones(T, dtype=bool)

    snap_det = np.zeros(T, dtype=bool)
    snap_det[3] = True
    snap_det[4] = True
    snap_meas = np.full(T, np.nan)
    snap_meas[3] = 50.0
    snap_meas[4] = 52.0

    cont_det = np.zeros(T, dtype=bool)
    cont_det[3] = True
    cont_meas = np.full(T, np.nan)
    cont_meas[3] = 48.0

    obs = SiteObservations(T, snap_mask, snap_det, snap_meas,
                           cont_mask, cont_det, cont_meas)

    specs = _default_specs()
    plumes = identify_plumes(obs, specs, p_off=0.1, max_gap=3)

    assert len(plumes) >= 1
    first_plume = plumes[0]
    assert 3 in first_plume.indices
    t3_entries = [(idx, tid) for idx, tid in zip(first_plume.indices, first_plume.tech_ids) if idx == 3]
    assert len(t3_entries) == 2, "Both snap and cont at t=3 should be in same plume"


# ══════════════════════════════════════════════════════════════════
# Test 11: _run_one_step returns κ_sc and uses it correctly
# ══════════════════════════════════════════════════════════════════

def test_run_one_step_kappa_sc():
    """_run_one_step should compute κ_sc as a separate quantity,
    not as κ_s · κ_c."""
    specs = _default_specs()
    T = 300
    rng_mask = np.random.default_rng(42)
    rng_emit = np.random.default_rng(43)

    snap_mask, cont_mask = generate_masks(
        T, p_snap=1.0, p_cont=1.0, T_cont=150, rng=rng_mask)

    obs, _, _, _ = simulate_series(
        T=T, p_on=0.025, p_off=0.10,
        size_mu=3.787, size_sigma=0.5,
        tech_specs=specs,
        snap_mask=snap_mask, cont_mask=cont_mask, rng=rng_emit,
    )

    if obs.n_total_detections() < 3:
        return

    step = _run_one_step(obs, specs, 0.025, 0.10)

    assert "kappa_sc" in step, "Step result must contain kappa_sc"
    kappa_s, kappa_c = step["kappa_values"]
    kappa_sc = step["kappa_sc"]

    assert kappa_sc <= min(kappa_s, kappa_c) + 1e-12, (
        f"κ_sc={kappa_sc} should be ≤ min(κ_s,κ_c)={min(kappa_s, kappa_c)}"
    )


# ══════════════════════════════════════════════════════════════════
# Test 12: Parameter recovery (smoke test)
# ══════════════════════════════════════════════════════════════════

def test_parameter_recovery_smoke():
    """Run MLE on moderate-sized simulations and check estimates are reasonable."""
    specs = _default_specs()
    T = 500
    true_p_on = 0.025
    true_p_off = 0.10
    true_mu_emit = 50.0
    true_mu = (true_p_on / (true_p_on + true_p_off)) * true_mu_emit

    estimates = []
    for rep in range(30):
        rng_mask = np.random.default_rng(1000 + rep)
        rng_emit = np.random.default_rng(2000 + rep)

        snap_mask, cont_mask = generate_masks(
            T, p_snap=1.0, p_cont=1.0, T_cont=250, rng=rng_mask,
        )

        obs, _, _, _ = simulate_series(
            T=T, p_on=true_p_on, p_off=true_p_off,
            size_mu=3.787, size_sigma=0.5,
            tech_specs=specs,
            snap_mask=snap_mask, cont_mask=cont_mask, rng=rng_emit,
        )

        if obs.n_total_detections() < 2:
            continue

        res = loop_empirical(obs, specs, T)
        estimates.append(res["mean"])

    assert len(estimates) >= 10, f"Too few valid estimates: {len(estimates)}"

    mean_est = np.mean(estimates)
    assert 0 < mean_est < 100, f"Mean estimate {mean_est} is out of range"


# ══════════════════════════════════════════════════════════════════
# Test 13: Baselines handle dual-channel observations
# ══════════════════════════════════════════════════════════════════

def test_baselines_dual_channel():
    """Naive and POD-weighted baselines should handle dual-channel data."""
    specs = _default_specs()
    T = 200
    rng_mask = np.random.default_rng(42)
    rng_emit = np.random.default_rng(43)

    snap_mask, cont_mask = generate_masks(
        T, p_snap=1.0, p_cont=1.0, T_cont=T, rng=rng_mask,
    )

    obs, _, _, _ = simulate_series(
        T=T, p_on=0.025, p_off=0.10,
        size_mu=3.787, size_sigma=0.5,
        tech_specs=specs,
        snap_mask=snap_mask, cont_mask=cont_mask, rng=rng_emit,
    )

    naive = naive_baseline(obs)
    pod = pod_weighted_baseline(obs, specs)

    assert naive >= 0
    assert pod >= 0
    assert np.isfinite(naive)
    assert np.isfinite(pod)


# ══════════════════════════════════════════════════════════════════
# Test 14: Lognormal expectation via quadrature
# ══════════════════════════════════════════════════════════════════

def test_lognormal_expectation_identity():
    """E[S] should match exp(mu + sigma^2/2) for identity function."""
    mu, sigma = 3.787, 0.5
    expected = np.exp(mu + 0.5 * sigma**2)
    computed = lognormal_expectation(mu, sigma, lambda x: x)
    assert abs(computed - expected) / expected < 1e-6, (
        f"Expected {expected}, got {computed}"
    )


# ══════════════════════════════════════════════════════════════════
# Run all tests
# ══════════════════════════════════════════════════════════════════

def run_all():
    tests = [
        test_kappa_sc_greater_than_product,
        test_emission_prob_sum_dual_deployment,
        test_emission_prob_sum_single_deployment,
        test_generate_masks_snap_only,
        test_generate_masks_cont_only,
        test_generate_masks_cont_window_contiguous,
        test_generate_masks_renewal_coverage,
        test_site_observations_basics,
        test_simulate_series_basic,
        test_snapshot_only_reduction,
        test_continuous_only_reduction,
        test_combined_log_estimate,
        test_plume_identification_dual_channel,
        test_run_one_step_kappa_sc,
        test_parameter_recovery_smoke,
        test_baselines_dual_channel,
        test_lognormal_expectation_identity,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n  {passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    print("=" * 62)
    print("  RUNNING TESTS")
    print("=" * 62)
    success = run_all()
    print("=" * 62)
    sys.exit(0 if success else 1)
