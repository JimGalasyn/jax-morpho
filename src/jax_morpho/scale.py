"""Neighbor-list (cell-list) relaxation for the center-based tissue engine.

The dense relaxation in ``center_based`` is O(N^2) and OOMs by ~30k cells.
This module computes the *same* Morse potential over a jax_md cell-list,
giving O(N) scaling that reaches 10^5 cells on CPU and (on GPU) the
million-cell organism regime.  Physics is identical to
``center_based.morse_energy`` — the neighbor list only skips pairs beyond
the cutoff, which contribute nothing anyway.

Kept separate from ``center_based`` so jax_md is an OPTIONAL dependency:
import this module only when you need scale.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space, partition

from jax_morpho.center_based import (
    D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT)


def pack_into_box(pos, margin_factor=3.0, r_max=R_MAX_DEFAULT):
    """Shift positions into [margin, extent+margin] and return (pos, box).

    A margin >= r_max on all sides means no cell is within the cutoff of a
    periodic image, so the periodic cell-list produces the same neighbors
    as free space.
    """
    pos = np.asarray(pos, np.float32)
    pos = pos - pos.min(0) + margin_factor * r_max
    box = float(pos.max() + margin_factor * r_max)
    return jnp.asarray(pos), box


def _morse_u(r, D, a, r_eq, r_max):
    raw = D * ((1.0 - jnp.exp(-a * (r - r_eq))) ** 2 - 1.0)
    shift_e = D * ((1.0 - jnp.exp(-a * (r_max - r_eq))) ** 2 - 1.0)
    return raw - shift_e


def build_neighbor_energy(box, D=D_DEFAULT, a=A_DEFAULT, r_eq=R_EQ_DEFAULT,
                          r_max=R_MAX_DEFAULT, dr_threshold=None,
                          capacity_multiplier=1.5):
    """Return (neighbor_fn, energy_fn, shift_fn) for the cell-list Morse
    energy on a periodic box of side ``box``."""
    if dr_threshold is None:
        dr_threshold = 0.3 * r_eq
    disp, shift = space.periodic(box)
    neighbor_fn = partition.neighbor_list(
        disp, box, r_cutoff=r_max, dr_threshold=dr_threshold,
        capacity_multiplier=capacity_multiplier, format=partition.Sparse)
    bond = space.map_bond(disp)

    def energy_fn(R, nbrs):
        i, j = nbrs.idx
        n = R.shape[0]
        # Sparse neighbor lists pad unused slots with the sentinel index n.
        # JAX clamps out-of-bounds gathers rather than erroring, but clamp
        # explicitly so the gather is unambiguous; the padded pairs are then
        # zeroed by the validity mask below.
        ic, jc = jnp.minimum(i, n - 1), jnp.minimum(j, n - 1)
        Rij = bond(R[ic], R[jc])
        r = jnp.sqrt((Rij * Rij).sum(-1) + 1e-9)
        valid = (j < n) & (r < r_max)
        u = jnp.where(valid, _morse_u(r, D, a, r_eq, r_max), 0.0)
        return 0.5 * u.sum()

    return neighbor_fn, energy_fn, shift


def relax_neighbor_list(pos, n_steps=200, lr=0.02, max_disp=0.1,
                        D=D_DEFAULT, a=A_DEFAULT, r_eq=R_EQ_DEFAULT,
                        r_max=R_MAX_DEFAULT, capacity_multiplier=1.5,
                        box=None):
    """Relax positions to mechanical equilibrium via cell-list Morse.

    Returns (relaxed_positions, did_overflow).  ``did_overflow`` True means
    the neighbor buffer was too small (raise capacity_multiplier); results
    are then unreliable.
    """
    if box is None:
        pos, box = pack_into_box(pos, r_max=r_max)
    else:
        pos = jnp.asarray(pos, jnp.float32)
    neighbor_fn, energy_fn, shift = build_neighbor_energy(
        box, D, a, r_eq, r_max, capacity_multiplier=capacity_multiplier)
    nbrs = neighbor_fn.allocate(pos)
    grad_fn = jax.grad(energy_fn)

    @jax.jit
    def run(R, nbrs):
        def step(carry, _):
            R, nbrs = carry
            g = grad_fn(R, nbrs)
            s = lr * g
            norm = jnp.sqrt((s * s).sum(-1, keepdims=True) + 1e-12)
            s = s * (max_disp / (norm + max_disp))
            R = shift(R, -s)
            nbrs = nbrs.update(R)
            return (R, nbrs), None
        (R, nbrs), _ = jax.lax.scan(step, (R, nbrs), None, length=n_steps)
        return R, nbrs

    R, nbrs = run(pos, nbrs)
    R.block_until_ready()
    return np.asarray(R), bool(nbrs.did_buffer_overflow)
