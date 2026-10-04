# Persistent-size IPW revision

The previous implementation is preserved in Git history at commit `8906acd`.
`source_manifest.json` remains the original collection record; current hashes
are in `package_manifest.json`. The original workspace has a separate backup.

The three-step estimator retains hard plume linking, log-space measurement
combination with the original lognormal correction, event-level backward-recursion
IPW, and the outer iteration. A frozen original-step fixture checks these outputs
at unit nudge. The simulator and original model tests are unchanged.

The transition update now conditions on the full weighted empirical rate law,
with OFF plus one ON state per corrected event size. Within-event persistence and
stop/restart transitions across observation gaps are represented explicitly.
The structured forward recursion takes O(observation times × empirical sizes).
It is exact conditional on the discrete rate law; estimating that law and linking
events are approximations. The algorithm is neither joint MLE nor exact EM.

The empirical nudge is removed: a nonunit compatibility argument raises an error.
The optimizer uses continuous log probabilities on fixed bounds independent of
simulation truth, with fixed starts and first-iteration coarse search. Constant
observation-count scaling only conditions the numerical objective. The unscaled
log likelihood is returned. Outer convergence and final numerical flags are
reported, and finite flagged estimates are retained. Zero-detection observed
campaigns contribute mean zero and unavailable component parameters.

The ungrouped comparator uses the same persistent transition likelihood with its
original per-detection weighted rate law, preserving the contrast with event-level
weighting, measurement combination and iteration. Legacy optional variants are
not used in the paper and retain their historical implementation.

Every required simulation uses the original seeds and nested sample counts.
Parallel workers preserve ordered results and independent campaign seeds.
Numerical sensitivity now varies optimizer tolerances, rather than a grid centred
on generating truth. Baseline and linking-threshold Monte Carlo bands use Student
t intervals. Misspecification MSE uses the actual batch squared errors for its
Monte Carlo SE and the POD MSE reference. Campaign planning counts use ceilings;
the bands propagate batch variance percentiles and assume negligible bias.

Manuscript text distinguishes conditional likelihood from the joint likelihood,
explains the empirical rate representation and computational cost, qualifies
asymptotic claims, and reports measured bias and numerical limitations. References,
full captions, figure annotations and the Word manuscript are synchronized with
the revised simulations. SI cross-reference labels are prefixed to avoid citation
collisions, and supplementary sections, equations, tables and figures have S
numbers. Multi-file compilation supports latexmk or Tectonic; the combined
Overleaf project retains cache-safe main/SI compilation settings.
