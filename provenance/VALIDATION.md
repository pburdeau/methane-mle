# Validation of persistent-size IPW revision

The previous nudge implementation remains in Git history at `8906acd`.
Its original validation records are in `previous_validation/`; they describe
that earlier method and must not be treated as checks of this revision.
The current scientific version is `persistent-ipw-v1`.

## Method and numerical verification

All 17 original model tests and six new estimator checks pass. New checks compare
the structured forward recursion with independently constructed dense matrix
powers for multiple support sizes, irregular observation gaps, and transition
probabilities whose sum exceeds one. They check within-event size dependence,
NumPy/C agreement, the original backward IPW calculation, nudge rejection,
numerical diagnostics, and zero-detection handling. A frozen fixture generated
from the original implementation verifies unchanged plume assignments, combined
rates, IPW weights, and weighted means at unit nudge.

Twelve selected baseline/sparse campaigns reproduce all ten checked estimator
fields, including the first zero-detection campaign in each scenario. Numerical
optimizer/backend tolerances are 0.005 kg/h for rates and 0.00001 for transitions,
with relative tolerance 0.00001 and matching missing component values.

Every required experiment is rerun with the original seeds, designs, and sample
counts. Baseline and sparse designs each contain 25 batches of 500 campaigns.
Other counts and current CSV hashes are recorded in `analysis_manifest.json`.
The numerical-accuracy experiment reuses the same 500 datasets at four optimizer
tolerances; these are not four independent datasets. Fresh outputs are saved in
`data/`, and all 13 computational figure assets are regenerated. The supplied
external schematic is copied unchanged. The unused legacy joint-perturbation CSV
is omitted; that original file remains in Git history.

The optimizer domain is fixed at [0.00001, 0.999999] for both transitions,
independent of simulation truth. The objective is the persistent-size detection
likelihood conditional on the full weighted empirical rate law. Nonunit nudge
arguments raise an error. Linking, measurement combination, event IPW and the
outer iteration remain; the law estimated from linked noisy rates is still a
plug-in approximation, not joint MLE or exact EM. These checks do not prove
unbiasedness, consistency, efficiency, or individual-record interval coverage.

## Measured results

Under baseline monitoring, MLE mean bias is +0.0766 kg/h on a true mean of
10 kg/h (Monte Carlo SE 0.0381; one-sample t=2.013, p=0.0555). Its 95% Student-t
Monte Carlo interval is approximately [-0.002, 0.155] kg/h. Non-rejection of zero
bias is not proof of unbiasedness. Average within-batch variance is 19.5831 for
MLE and 29.6350 for POD weighting, a 33.9192% reduction. Ungrouped variance is
20.6412, so the grouped procedure improves by 5.1% relative to that comparator.
The comparator changes event linking, averaging, event-level IPW and iteration
together; it does not isolate linking alone.

In the sparse design, MLE bias is +0.0436 kg/h (p=0.2869). Variances are 27.0083
and 31.9409 for MLE and POD weighting, a 15.4429% reduction. Both scenarios retain
all 12,500 mean estimates, including two baseline and 16 sparse zero-detection
campaigns assigned zero. Component parameters remain missing for those records.
Among records with detections, 57 baseline and 45 sparse campaigns reach the
outer iteration limit; two and five have final optimizer flags. Two baseline and
18 sparse estimates reach a transition boundary. Finite flagged estimates remain
in the summaries. The mean baseline OFF-transition estimate is 0.05851 versus
0.05 truth; residual component bias is reported in the manuscript.

Baseline planning counts, rounded upward, are 836 versus 1265 campaigns at 3%
precision, 301 versus 456 at 5%, and 76 versus 114 at 10%. These normal-approximation
counts assume independent campaigns and negligible bias at the target precision.
Their bands propagate batch variance percentiles, rather than confidence intervals
for the average planning count. A paired before/after design additionally requires
the covariance of repeated measurements.

Misspecification MSE combines within-batch variance, between-batch variability,
and squared overall bias. Its Monte Carlo standard error is calculated directly
from batch mean squared errors. The POD reference is its measured MSE, rather
than just its variance. Large-calibration-error comparisons are qualified by
the broad Monte Carlo bands. Curves use the original 25 calibration draws per
setting; finite simulation noise need not yield monotone sensitivity curves.

## Manuscripts and submission artifacts

The main paper and SI compile with Tectonic 0.16.9 using their local figures and
bibliographies. Logs have no undefined citations/references, missing glyphs, or
overfull boxes. The compiler emits package-encoding and PostScript crop-special
warnings from the supplied OUP template; the rendered PDFs are checked separately.
Switching one shared `output.aux` from main to SI and back succeeds. The SI loads
lineno without enabling line numbers and hyperref to read the other document's
cached commands. Imported SI labels have an `SI-` prefix; supplementary sections,
equations, tables and figures use S numbering. The combined upload ZIP contains
exactly two root TeX documents and the 14 referenced external PDF figure assets.

The revised Word main manuscript has 37 rendered pages, six figures, two editable
tables and 11 numbered native equations. Full source captions are retained,
body text is justified, and line numbering is continuous after the title page.
Every rendered page is visually checked, including mathematical displays and
figure/caption placement. Existing funding, affiliations and author declarations
are retained. The supplied Reuland 2025 reference is included, and the Conrad
preprint title placeholder is completed from the publisher's record.

## Evidence and environment

`validation/` contains current test, replay, statistics, figure, compilation and
cache-switch records. The generated Word document and combined Overleaf archive
are provided separately in the parent workspace. `source_manifest.json` preserves
the original collection record; `package_manifest.json` verifies this revision's
files. Raw working outputs under `build/` and compiled native libraries are ignored.

Verified with Python 3.14.5, NumPy 2.3.4, pandas 2.3.3, Matplotlib 3.10.7 and
SciPy 1.16.2. The optional C accelerator implements the same recursion as NumPy;
no platform-specific library is committed. PDF metadata/fonts and numerical
optimizer output can vary across platforms. The authors should choose a license
for the public code/data repository before the paper is submitted.
