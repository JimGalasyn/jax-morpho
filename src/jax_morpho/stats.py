"""Epithelial topology statistics and reference distributions.

A 2D cellular tissue's polygon-side distribution is a standard calibration
target: a random (Poisson–Voronoi) point pattern and a mechanically ordered
proliferating epithelium (Gibson 2006) have distinctly different side-count
histograms. The center-based engine measures its own distribution via
``jax_morpho.center_based.interior_side_counts`` (Delaunay degree); the
references and distance here turn "looks plausible" into a number.

References
----------
Gibson, M.C. et al. (2006). Nature 442, 1038 — proliferating epithelia.
Poisson-Voronoi side distribution: standard 2D stochastic geometry.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Reference distributions (fraction of interior cells with k sides)
# ---------------------------------------------------------------------------

#: 2D Poisson-Voronoi tessellation — the null "random seed cloud" model.
POISSON_VORONOI_SIDES: dict[int, float] = {
    3: 0.0113, 4: 0.1068, 5: 0.2595, 6: 0.2949,
    7: 0.1988, 8: 0.0898, 9: 0.0330, 10: 0.0104,
}

#: Real proliferating epithelium (Gibson 2006, approx pooled Drosophila /
#: Hydra / Xenopus) — hexagon-dominated, tighter than Poisson-Voronoi.
GIBSON_EPITHELIUM_SIDES: dict[int, float] = {
    4: 0.03, 5: 0.25, 6: 0.45, 7: 0.20, 8: 0.05, 9: 0.01,
}


def side_distribution(side_counts, ks: range = range(3, 11)) -> dict[int, float]:
    """Fraction of cells with k sides, from an array of per-cell side counts
    (e.g. the output of ``center_based.interior_side_counts``)."""
    counts = np.asarray(side_counts)
    n = max(len(counts), 1)
    return {k: float((counts == k).sum()) / n for k in ks}


def mean_sides(side_counts) -> float:
    """Mean side count (Euler predicts -> 6 for a 2D tessellation)."""
    return float(np.asarray(side_counts).mean())


def l1_distance(dist: dict[int, float], reference: dict[int, float]) -> float:
    """Total-variation-style L1 distance between two side distributions,
    summed over the UNION of their supports so mass in either that falls
    outside the other's keys is not silently dropped (keeps the metric
    symmetric)."""
    keys = set(dist) | set(reference)
    return float(sum(abs(dist.get(k, 0.0) - reference.get(k, 0.0))
                     for k in keys))
