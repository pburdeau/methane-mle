"""
Centralised baseline parameters for the methane MLE paper.

Hybrid observation model:
  - Snapshot: per-time-step Bernoulli(p_snap).
  - Continuous: at each unmonitored step, Bernoulli(p_cont) triggers
    deployment of a monitor that runs for T_cont contiguous steps.
"""

import numpy as np

# ── Emission process ──────────────────────────────────────────────
T = 1000                    # Campaign duration (time steps / hours)
MU_EMIT = 50.0             # Mean emission rate when ON (kg/h)
TAU_EMIT = 20.0            # Mean ON duration (hours)
PI_ON = 0.2                # Stationary ON probability

P_OFF = 1.0 / TAU_EMIT
P_ON = PI_ON * P_OFF / (1.0 - PI_ON)

SIZE_MU = 3.787            # Lognormal location (log-scale)
SIZE_SIGMA = 0.5           # Lognormal scale    (log-scale)

# ── Observation process (hybrid model) ───────────────────────────
P_SNAP = 0.125             # Per-time-step prob. of snapshot observation
P_CONT = 0.01              # Per-step prob. of starting continuous window
T_CONT = 50                # Duration of continuous monitor (time steps)

# ── Aerial / snapshot sensor ─────────────────────────────────────
AERIAL_THRESHOLD = 50.0    # 50 % POD threshold (kg/h)
AERIAL_SLOPE = 2.0
AERIAL_SENSOR_SIGMA = 0.05

# ── Continuous sensor ────────────────────────────────────────────
CONT_THRESHOLD = 2.403749283845681     # 50 % POD threshold (kg/h)
CONT_SLOPE = 3.0
CONT_SENSOR_SIGMA = 0.2
CONT_FP_RATE = 0.0
CONT_FP_SCALE = 4.0

# ── Algorithm ────────────────────────────────────────────────────
MAX_GAP = 3
DECISION_THRESHOLD = 0.5   # Bayesian plume-linking threshold
BASE_SEED = 2602
NUDGE = 1.0                # Compatibility only: no likelihood adjustment.
TRANSITION_BOUNDS = (1e-5, 1-1e-6)
MAX_ITER = 30
TRANSITION_TOL = 1e-5
MEAN_TOL = 1e-4
METHOD_VERSION = "persistent-ipw-v1"
MLE_OPTIONS = dict(transition_bounds=TRANSITION_BOUNDS, max_iter=MAX_ITER,
                   tol=TRANSITION_TOL, mean_tol=MEAN_TOL)

# ── Legacy grid helper, retained for explicit numerical diagnostics ────────────────────────────
P_GRID_RES = 11            # Number of grid points (must be odd)
P_GEOM_FACTOR = 1.2       # Geometric ratio between consecutive points


def build_geom_grid(center, factor, n):
    """Build a geometric grid of *n* points centered on *center*."""
    assert n % 2 == 1, "P_GRID_RES must be odd"
    half = n // 2
    return center * factor ** np.arange(-half, half + 1)
    # half = n // 2
    # return center + np.arange(-half, half + 1) * center / half / 2


# ── Derived / convenience ───────────────────────────────────────
TRUE_PARAMS = {
    "p_on":    P_ON,
    "p_off":   P_OFF,
    "mu_emit": MU_EMIT,
    "mu":      PI_ON * MU_EMIT,
}
