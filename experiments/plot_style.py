"""Shared figure colours for the main manuscript and supplementary information.

Method colours draw on Paul Tol's vibrant qualitative scheme:
https://sronpersonalpages.nl/~pault/#sec:qualitative
Keep each estimator's colour consistent across all comparisons.
"""

NAIVE = "#999999"
POD = "#EE7733"
UNGROUPED = "#0077BB"
MLE = "#009988"

SNAPSHOT = UNGROUPED
CONTINUOUS = POD
TRUTH = "#333333"
RATIO = "#555555"
EMISSION_BAND = "#EEEEBB"
EMISSION_SIZE = "#AA3377"
TOTAL_ERROR = "#333333"
PANEL_BACKGROUND = "#F1F7F6"

# Parameter panels distinguish the two transition probabilities, the active
# emission rate, and the time-averaged rate; the latter uses the MLE colour.
PARAMETERS = (UNGROUPED, POD, EMISSION_SIZE, MLE)
