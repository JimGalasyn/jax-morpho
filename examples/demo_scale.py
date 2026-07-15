"""Neighbor-list scaling demo for the center-based tissue engine.

Shows that computing the same Morse potential over a jax_md cell-list
(1) reaches the same energy minimum as the dense O(N^2) relaxation and
(2) scales to cell counts the dense version can't hold.

Needs the optional 'scale' extra:  pip install -e '.[scale]'
Run:  .venv/bin/python examples/demo_center_based_scale.py
"""
from __future__ import annotations

import time

import numpy as np
import jax.numpy as jnp

from jax_morpho.center_based import (
    D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT, morse_energy, relax)
from jax_morpho.scale import (
    pack_into_box, relax_neighbor_list)


def grid_blob(N, spacing=1.0, noise=0.15, seed=0):
    side = int(np.ceil(np.sqrt(N)))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    pts = np.column_stack([xs.ravel(), ys.ravel()])[:N].astype(np.float32) * spacing
    rng = np.random.default_rng(seed)
    return pts + rng.normal(0, noise, pts.shape).astype(np.float32)


def dense_energy(pos):
    p = jnp.asarray(pos, jnp.float32)
    return float(morse_energy(p, jnp.ones(len(pos), jnp.float32),
                              D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT))


def main():
    print("=== CORRECTNESS: dense vs neighbor-list reach the same minimum ===")
    pos = grid_blob(1000, seed=1)
    p = jnp.asarray(pos, jnp.float32)
    alive = jnp.ones(len(pos), jnp.float32)
    dense_final = relax(p, alive, D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT,
                        R_MAX_DEFAULT, 200)
    nl_final, overflow = relax_neighbor_list(pos, n_steps=200)
    print(f"  dense minimum         = {dense_energy(np.asarray(dense_final)):.3f}")
    print(f"  neighbor-list minimum = {dense_energy(nl_final):.3f}")
    print(f"  overflow: {overflow}")

    print("\n=== SCALING: ms per relaxation step vs N ===")
    print(f"  {'N':>7} {'dense ms':>10} {'nbrlist ms':>11} {'speedup':>9}")
    for N in (1000, 4000, 16000, 64000, 150000):
        pos = grid_blob(N, seed=2)
        dense_ms = None
        if N <= 16000:
            p = jnp.asarray(pos, jnp.float32)
            al = jnp.ones(N, jnp.float32)
            relax(p, al, D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT, 1).block_until_ready()
            t0 = time.time()
            relax(p, al, D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT, 20).block_until_ready()
            dense_ms = (time.time() - t0) / 20 * 1e3
        pb, box = pack_into_box(pos)
        relax_neighbor_list(np.asarray(pb), n_steps=1, box=box)  # warm
        t0 = time.time()
        _, of = relax_neighbor_list(np.asarray(pb), n_steps=20, box=box)
        nl_ms = (time.time() - t0) / 20 * 1e3
        d = f"{dense_ms:10.1f}" if dense_ms else f"{'OOM/skip':>10}"
        sp = f"{dense_ms/nl_ms:8.1f}x" if dense_ms else f"{'-':>9}"
        print(f"  {N:>7} {d} {nl_ms:11.1f} {sp}{'  [OVERFLOW]' if of else ''}")
    print("  (CPU float32; GPU + this same cell-list -> the million-cell regime)")


if __name__ == "__main__":
    main()
