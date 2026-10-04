"""Independent numerical and compatibility checks for the likelihood repair."""
import json
from pathlib import Path

import numpy as np

from src.persistent import (build_native, forward_loglik, detection_inputs,
                            event_detection_probability)
from src.estimate import _run_one_step, loop_empirical
from src import build_baseline_specs, generate_masks, simulate_series


def test_forward_against_dense_transition_powers():
    rng = np.random.default_rng(124)
    for k in (1,3,8):
        for a,b in ((.0125,.05),(.7,.8)):
            w = rng.dirichlet(np.ones(k))
            transition = np.zeros((k+1,k+1))
            transition[0,0], transition[0,1:] = 1-a,a*w
            transition[1:,0] = b
            transition[1:,1:] = (1-b)*np.eye(k)
            on = np.ascontiguousarray(rng.uniform(.02,.95,(7,k)))
            off = np.ascontiguousarray(rng.uniform(.02,.95,7))
            gaps = np.array([1.,2.,3.,8.,4.,1.])
            alpha = np.r_[b/(a+b),a/(a+b)*w]*np.r_[off[0],on[0]]
            for t,g in enumerate(gaps,1):
                alpha = (alpha@np.linalg.matrix_power(transition,int(g)))*np.r_[off[t],on[t]]
            expected = np.log(alpha.sum())
            for backend in ('numpy','auto'):
                np.testing.assert_allclose(forward_loglik(a,b,w,(on,off,gaps),backend),expected,atol=1e-12)


def test_same_event_dependence_is_retained():
    # Conditioning on an ON start, two consecutive detections include a
    # same-event contribution proportional to E[q^2], not E[q]^2.
    w = np.array([.5,.5]); q = np.array([.1,.9])
    a,b = .02,.1; pi1 = a/(a+b)
    on = np.ascontiguousarray(np.tile(q,(2,1)))
    off = np.zeros(2); gaps = np.ones(1)
    expected = pi1*(1-b)*np.dot(w,q*q)
    actual = np.exp(forward_loglik(a,b,w,(on,off,gaps)))
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    assert actual > pi1*(1-b)*np.dot(w,q)**2


def test_event_ipw_acceleration_matches_original_recursion():
    miss = np.random.default_rng(6).uniform(.01,1,(29,60))
    for b in (.005,.05,.95):
        alpha = np.ones(60); total = np.zeros(60)
        for row in miss[::-1]:
            alpha = row*(b+(1-b)*alpha); total += alpha
        np.testing.assert_allclose(event_detection_probability(b,miss),1-total/len(miss),atol=1e-14)


def example():
    specs = build_baseline_specs(50,2,.05,2.403749283845681,3,.2,0)
    sm,cm = generate_masks(1000,.125,.01,50,np.random.default_rng(2602))
    obs,*_ = simulate_series(1000,.0125,.05,3.787,.5,specs,sm,cm,np.random.default_rng(102602))
    return obs,specs


def test_nudge_rejected_and_diagnostics_returned():
    obs,specs = example()
    try:
        loop_empirical(obs,specs,obs.T,p_off_nudge=1.015)
    except ValueError:
        pass
    else:
        raise AssertionError('A nonunit likelihood nudge must be rejected.')
    result = loop_empirical(obs,specs,obs.T)
    assert np.isfinite(result['mean'])
    assert result['n_iterations'] <= 30
    assert {'optimizer_success','has_detections','boundary','empirical_weights'} <= result.keys()


def test_zero_detections_retained_without_fabricated_parameters():
    obs,specs = example()
    obs.snap_detected[:] = False; obs.cont_detected[:] = False
    obs.snap_measurements[:] = np.nan; obs.cont_measurements[:] = np.nan
    result = loop_empirical(obs,specs,obs.T)
    assert result['mean'] == 0 and not result['has_detections']
    assert np.isnan(result['p_on']) and np.isnan(result['mu_emit'])


def test_original_linking_and_ipw_unchanged():
    fixture = json.loads((Path(__file__).parent / 'fixtures/preserved_plume_step.json').read_text())
    obs,specs = example()
    new = _run_one_step(obs,specs,.025,.1)
    for key in ('plume_sizes','ipw_weights','weighted_mean','kappa_values','kappa_sc'):
        np.testing.assert_allclose(new[key],fixture[key],rtol=1e-13,atol=1e-13)
    assert [p.indices for p in new['plumes']] == fixture['plume_indices']


if __name__ == '__main__':
    try:
        build_native()
    except RuntimeError:
        print('Native acceleration unavailable; testing NumPy fallback.')
    for name,fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('PASS',name)
