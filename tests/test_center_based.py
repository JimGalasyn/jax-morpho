"""Calibration tests for the center-based differentiable tissue engine.

Encodes the emergent result that coupling proliferation to relaxation
reproduces the real-epithelium (Gibson 2006) polygon-side distribution,
that the balance is a tunable ordering knob, and that the relaxation is
autodiff-differentiable end-to-end (the inverse-design hook).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jax_morpho.center_based import (
    D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT,
    grow_relax, interior_side_counts, morse_energy, relax,
)
from jax_morpho.stats import (
    GIBSON_EPITHELIUM_SIDES, POISSON_VORONOI_SIDES, l1_distance,
)


def _dist(sides):
    # Cover the full support (3..10) that the reference distributions span,
    # so no cell mass is dropped when comparing (avoids biasing the
    # PV-vs-Gibson distance, which now sums over the union of keys).
    n = max(len(sides), 1)
    return {k: float((sides == k).sum()) / n for k in range(3, 11)}


@pytest.fixture(scope="module")
def proliferation_dominated():
    """Small relaxation per division -> should land near Gibson."""
    pos, alive = grow_relax(n_max=380, n_start=15, target=280,
                            relax_steps=8, seed=1)
    return interior_side_counts(pos, alive)


@pytest.fixture(scope="module")
def relaxation_dominated():
    """Heavy relaxation per division -> should over-order (crystalline)."""
    pos, alive = grow_relax(n_max=380, n_start=15, target=280,
                            relax_steps=80, seed=1)
    return interior_side_counts(pos, alive)


class TestRelaxationValidity:
    def test_relax_lowers_energy(self):
        rng = np.random.default_rng(0)
        pos = jnp.asarray(rng.normal(0, 6, (120, 2)).astype(np.float32))
        alive = jnp.ones(120, np.float32)
        args = (alive, D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT)
        e0 = float(morse_energy(pos, *args))
        pos2 = relax(pos, *args, 300)
        e1 = float(morse_energy(pos2, *args))
        assert e1 < e0, f"relaxation must lower energy: {e0:.2f} -> {e1:.2f}"


class TestGibsonCalibration:
    def test_mean_sides_near_six(self, proliferation_dominated):
        m = proliferation_dominated.mean()
        assert 5.6 <= m <= 6.15, f"mean sides {m:.3f} should be near 6"

    def test_matches_real_epithelium(self, proliferation_dominated):
        d = _dist(proliferation_dominated)
        hexfrac = d[6]
        assert 0.38 <= hexfrac <= 0.62, (
            f"hexagon fraction {hexfrac:.3f} should sit in the real-epithelium "
            f"range, not random (~0.29) or crystalline (>0.7)")
        # Closer to Gibson than to the random Poisson-Voronoi baseline.
        assert (l1_distance(d, GIBSON_EPITHELIUM_SIDES)
                < l1_distance(d, POISSON_VORONOI_SIDES)), (
            "proliferation+relaxation should match real epithelium better "
            "than a random tessellation")

    def test_relaxation_is_an_ordering_knob(self, proliferation_dominated,
                                            relaxation_dominated):
        h_lo = _dist(proliferation_dominated)[6]
        h_hi = _dist(relaxation_dominated)[6]
        assert h_hi > h_lo + 0.03, (
            f"more relaxation per division must increase order: "
            f"hex {h_lo:.3f} (prolif) vs {h_hi:.3f} (relax)")


class TestDifferentiability:
    def test_gradient_flows_through_relaxation(self):
        rng = np.random.default_rng(2)
        p0 = jnp.asarray(rng.normal(0, 4, (150, 2)).astype(np.float32))
        alive = jnp.ones(150, np.float32)

        def observable(D):
            p = relax(p0, alive, D, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT, 150)
            diff = p[:, None, :] - p[None, :, :]
            r2 = (diff * diff).sum(-1) + jnp.eye(150) * 1e9
            return jnp.sqrt(r2.min(axis=1) + 1e-12).mean()

        g = float(jax.grad(observable)(1.0))
        assert np.isfinite(g), f"gradient must be finite, got {g}"
        assert g != 0.0, "gradient should be nonzero (observable depends on D)"
