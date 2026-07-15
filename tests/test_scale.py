"""Calibration tests for the neighbor-list (cell-list) center-based relaxer.

The neighbor-list relaxation must (1) reach the same energy minimum as the
dense O(N^2) relaxation — same physics — and (2) run at cell counts the
dense version can't (no buffer overflow), which is the scaling win.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

jax_md = pytest.importorskip("jax_md")  # optional dependency

from jax_morpho.center_based import (
    D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT,
    morse_energy, relax,
)
from jax_morpho.scale import (
    pack_into_box, relax_neighbor_list,
)


def _grid_blob(N, spacing=1.0, noise=0.15, seed=0):
    side = int(np.ceil(np.sqrt(N)))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    pts = np.column_stack([xs.ravel(), ys.ravel()])[:N].astype(np.float32) * spacing
    rng = np.random.default_rng(seed)
    return pts + rng.normal(0, noise, pts.shape).astype(np.float32)


def _dense_energy(pos):
    p = jnp.asarray(pos, jnp.float32)
    return float(morse_energy(p, jnp.ones(len(pos), jnp.float32),
                              D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT))


class TestNeighborListMatchesDense:
    def test_same_energy_minimum(self):
        pos = _grid_blob(900, seed=1)
        # dense relaxation
        p = jnp.asarray(pos, jnp.float32)
        alive = jnp.ones(len(pos), jnp.float32)
        dense_final = relax(p, alive, D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT,
                            R_MAX_DEFAULT, 200)
        e_dense = _dense_energy(np.asarray(dense_final))
        # neighbor-list relaxation of the same start config
        nl_final, overflow = relax_neighbor_list(pos, n_steps=200)
        e_nl = _dense_energy(nl_final)
        assert not overflow, "neighbor buffer overflowed"
        # Both reach essentially the same minimum (translation aside, energy
        # is frame-invariant).
        assert abs(e_dense - e_nl) < 0.02 * abs(e_dense) + 1.0, (
            f"neighbor-list minimum {e_nl:.1f} should match dense {e_dense:.1f}")

    def test_relaxation_lowers_energy(self):
        pos = _grid_blob(900, seed=3)
        e0 = _dense_energy(pack_into_box(pos)[0])
        final, overflow = relax_neighbor_list(pos, n_steps=200)
        assert not overflow
        assert _dense_energy(final) < e0


class TestScale:
    def test_handles_large_N_without_overflow(self):
        # 8000 cells: dense is ~250MB+ per pairwise array and slow; the
        # cell-list handles it in a few seconds with no overflow.
        pos = _grid_blob(8000, seed=5)
        final, overflow = relax_neighbor_list(pos, n_steps=50)
        assert not overflow, "raise capacity_multiplier if this overflows"
        assert final.shape == (8000, 2)
        assert np.isfinite(final).all()
