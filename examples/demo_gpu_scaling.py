"""GPU scaling of the center-based neighbor-list relaxation.

Runs the same cell-list Morse relaxation used by center_based_scale at up
to a million+ cells and reports throughput.  On an RTX 4090 this reaches
1e6 cells at ~32 ms/step (~31M cell-updates/s), peak ~5 GB -- the actual
millions-of-cells regime, not an extrapolation.

Needs the 'scale' extra and a CUDA-enabled jax:
    pip install -e '.[scale]'  &&  pip install 'jax[cuda12]'
Run:  .venv/bin/python examples/demo_gpu_scaling.py
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from jax_morpho.center_based import R_MAX_DEFAULT as r_max, R_EQ_DEFAULT as r_eq
from jax_morpho.scale import build_neighbor_energy, pack_into_box


def grid_blob(N, spacing=1.0, noise=0.15, seed=0):
    side = int(np.ceil(np.sqrt(N)))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    pts = np.column_stack([xs.ravel(), ys.ravel()])[:N].astype(np.float32) * spacing
    rng = np.random.default_rng(seed)
    return pts + rng.normal(0, noise, pts.shape).astype(np.float32)


def bench(N):
    pos, box = pack_into_box(grid_blob(N, seed=2))
    neighbor_fn, energy_fn, shift = build_neighbor_energy(box)
    nbrs = neighbor_fn.allocate(pos)
    grad_fn = jax.grad(energy_fn)

    @partial(jax.jit, static_argnums=(2,))
    def run(R, nbrs, n):
        def step(carry, _):
            R, nbrs = carry
            g = grad_fn(R, nbrs)
            s = 0.02 * g
            norm = jnp.sqrt((s * s).sum(-1, keepdims=True) + 1e-12)
            s = s * (0.1 / (norm + 0.1))
            R = shift(R, -s)
            nbrs = nbrs.update(R)
            return (R, nbrs), None
        (R, nbrs), _ = jax.lax.scan(step, (R, nbrs), None, length=n)
        return R, nbrs

    R, nbrs = run(pos, nbrs, 1)          # warm compile
    R.block_until_ready()
    t0 = time.time()
    R, nbrs = run(R, nbrs, 20)
    R.block_until_ready()
    return (time.time() - t0) / 20, bool(nbrs.did_buffer_overflow)


def main():
    print("jax backend:", jax.default_backend(), jax.devices())
    print(f"\n  {'N':>9} {'ms/step':>9} {'cell-updates/s':>16}  status")
    for N in (10_000, 100_000, 1_000_000, 2_000_000):
        try:
            dt, of = bench(N)
            print(f"  {N:>9} {dt*1e3:9.1f} {N/dt/1e6:13.1f}M   "
                  f"{'OVERFLOW' if of else 'ok'}")
        except Exception as e:
            print(f"  {N:>9} {'-':>9} {'-':>16}   FAILED: {type(e).__name__}")
            break
    try:
        st = jax.devices()[0].memory_stats()
        print(f"\n  peak GPU memory: {st.get('peak_bytes_in_use', 0)/1e9:.2f} GB")
    except Exception:
        pass


if __name__ == "__main__":
    main()
