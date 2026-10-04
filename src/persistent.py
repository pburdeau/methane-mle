"""Detection likelihood conditional on a weighted empirical event-size law.

An ON event retains its size until it ends. The finite-support recursion is
exact conditional on that law; estimating the law from linked, noisy plumes
is a plug-in approximation, not joint maximum likelihood or an EM algorithm.
The optional C implementation accelerates the same recursion. NumPy is the
portable fallback; run ``python -m src.persistent --build-native`` to build C.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize

_NATIVE = None
_ATTEMPTED = False


def _library_path():
    source = Path(__file__).with_name('persistent.c')
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return source.parent / '.native' / f'persistent_{digest}{suffix}'


def build_native():
    """Build the optional acceleration library with the local C compiler."""
    if sys.platform == 'win32':
        raise RuntimeError('Use the NumPy backend on Windows.')
    target = _library_path()
    if not target.exists():
        compiler = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
        if compiler is None:
            raise RuntimeError('No C compiler found; the NumPy backend remains available.')
        target.parent.mkdir(exist_ok=True)
        flag = '-dynamiclib' if sys.platform == 'darwin' else '-shared'
        try:
            subprocess.run([compiler, '-O3', '-fPIC', flag,
                            str(Path(__file__).with_name('persistent.c')), '-lm',
                            '-o', str(target)], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            target.unlink(missing_ok=True)
            raise RuntimeError('C acceleration could not be built; using NumPy.') from e
    global _ATTEMPTED
    _ATTEMPTED = False
    _native()
    return target


def _native():
    global _NATIVE, _ATTEMPTED
    if not _ATTEMPTED:
        _ATTEMPTED = True
        target = _library_path()
        if target.exists():
            try:
                lib = ctypes.CDLL(str(target))
                ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
                lib.forward_ll.argtypes = [ctypes.c_double, ctypes.c_double,
                    ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int]
                lib.forward_ll.restype = ctypes.c_double
                lib.event_detection.argtypes = [ctypes.c_double, ptr,
                    ctypes.c_int, ctypes.c_int, ptr]
                lib.event_detection.restype = None
                _NATIVE = lib
            except OSError:
                _NATIVE = None
    return _NATIVE


def detection_inputs(obs, tech_specs, sizes):
    """Observation probabilities for OFF and each persistent ON size."""
    times = obs.observed_times()
    sizes = np.asarray(sizes, dtype=float)
    on = np.ones((len(times), len(sizes)))
    off = np.ones(len(times))
    for spec, mask, detected in zip(tech_specs,
            (obs.snap_mask, obs.cont_mask), (obs.snap_detected, obs.cont_detected)):
        deployed, hit = mask[times], detected[times]
        q = np.asarray(spec.pod.probability(sizes))
        on *= np.where(deployed[:, None], np.where(hit[:, None], q, 1-q), 1)
        fp = spec.false_positive_rate
        off *= np.where(deployed, np.where(hit, fp, 1-fp), 1)
    return (np.ascontiguousarray(on), np.ascontiguousarray(off),
            np.ascontiguousarray(np.diff(times), dtype=float))


def forward_loglik(p_on, p_off, weights, inputs, backend='auto'):
    """Scaled forward likelihood, including stop/restart events within gaps.

    ``inputs`` contains N-by-K ON probabilities, N OFF probabilities, and
    N-1 positive integer gaps. Initial state probabilities are stationary.
    """
    on, off, gaps = inputs
    w = np.ascontiguousarray(weights, dtype=float)
    if not (0 < p_on < 1 and 0 < p_off < 1):
        return -np.inf
    if len(off) == 0:
        return 0.0
    lib = _native() if backend != 'numpy' else None
    if backend == 'native' and lib is None:
        raise RuntimeError('Native library unavailable; build it or select NumPy.')
    if lib is not None:
        return float(lib.forward_ll(p_on, p_off, w, on, off, gaps, len(off), len(w)))
    pi1 = p_on / (p_on+p_off)
    pi0 = 1-pi1
    r = 1-p_on-p_off
    a0 = pi0 * off[0]
    a = pi1*w*on[0]
    scale = a0+a.sum()
    if scale <= 0 or not np.isfinite(scale):
        return -np.inf
    result = np.log(scale)
    a0 /= scale
    a /= scale
    rn = np.power(r, gaps)
    stay = np.power(1-p_off, gaps)
    for t in range(1, len(off)):
        total = a.sum()
        t00 = pi0+pi1*rn[t-1]
        t01 = pi1*(1-rn[t-1])
        t10 = pi0*(1-rn[t-1])
        t11 = pi1+pi0*rn[t-1]
        reset = max(0.0, t11-stay[t-1])  # nonnegative analytically
        b0 = off[t]*(a0*t00+total*t10)
        a = on[t]*(a*stay[t-1]+w*(a0*t01+total*reset))
        scale = b0+a.sum()
        if scale <= 0 or not np.isfinite(scale):
            return -np.inf
        a0 = b0/scale
        a /= scale
        result += np.log(scale)
    return float(result)


def event_detection_probability(p_off, miss):
    """Original finite-window backward recursion, optionally accelerated."""
    miss = np.ascontiguousarray(miss, dtype=float)
    lib = _native()
    if lib is not None:
        result = np.empty(miss.shape[1])
        lib.event_detection(p_off, miss, *miss.shape, result)
        return result
    alpha = np.ones(miss.shape[1])
    total = np.zeros_like(alpha)
    for row in miss[::-1]:
        alpha = row*(p_off+(1-p_off)*alpha)
        total += alpha
    return 1-total/len(miss)


def fit_transitions(weights, inputs, start=(.025, .1), bounds=(1e-5, 1-1e-6),
                    coarse=False, p_on_grid=None, p_off_grid=None,
                    optimizer_tolerance=1e-11):
    """Conditional optimization on a fixed domain, independent of truth.

    Explicit grids are supported for numerical diagnostics only. The default
    uses continuous optimization and a fixed coarse start search on the first
    outer iteration. A failed optimizer is reported, never silently filtered.
    """
    lo, hi = bounds
    if not (0 < lo < hi < 1):
        raise ValueError('Transition bounds must satisfy 0 < lower < upper < 1.')
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0 or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError('Event-size weights must be finite, nonnegative, and nonempty.')
    w = np.ascontiguousarray(w/w.sum())
    on, off, gaps = inputs
    if on.shape != (len(off), len(w)) or len(gaps) != max(0, len(off)-1):
        raise ValueError('Inconsistent likelihood array dimensions.')
    if np.any(gaps < 1) or np.any(gaps != np.floor(gaps)):
        raise ValueError('Observation gaps must be positive integers.')
    if (p_on_grid is None) != (p_off_grid is None):
        raise ValueError('Supply both diagnostic grids or neither.')
    def objective(x):
        value = -forward_loglik(*np.exp(x), w, inputs)/max(1,len(off))
        return value if np.isfinite(value) else 1e100
    if p_on_grid is not None:
        candidates = [(a,b) for a in p_on_grid for b in p_off_grid if lo <= a <= hi and lo <= b <= hi]
        if not candidates:
            raise ValueError('No diagnostic grid values lie within transition bounds.')
        p = min(candidates, key=lambda x: objective(np.log(x)))
        value = forward_loglik(*p,w,inputs)
        return np.array(p), value, np.isfinite(value)
    starts = [np.clip(start,lo,hi)]
    if coarse:
        grid = np.clip([.0001,.001,.005,.015,.04,.1,.3,.8],lo,hi)
        candidates = [np.array([a,b]) for a in grid for b in grid]
        starts.append(min(candidates,key=lambda p:objective(np.log(p))))
    log_bounds = [(np.log(lo),np.log(hi))]*2
    results = [minimize(objective,np.log(p),method='L-BFGS-B',bounds=log_bounds,
        options={'ftol':optimizer_tolerance,'gtol':1e-8,'maxiter':150}) for p in starts]
    best = min(results,key=lambda x:x.fun)
    if not best.success or np.linalg.norm(best.jac,np.inf)>1e-5:
        retry = minimize(objective,best.x,method='Powell',bounds=log_bounds,
            options={'ftol':1e-10,'xtol':1e-7,'maxiter':120})
        if retry.fun < best.fun:
            best = retry
    return np.exp(best.x), -float(best.fun)*max(1,len(off)), bool(best.success and best.fun < 1e99)


if __name__ == '__main__':
    if sys.argv[1:] == ['--build-native']:
        print(build_native())
    else:
        raise SystemExit('Usage: python -m src.persistent --build-native')
