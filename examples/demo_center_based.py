"""Center-based differentiable tissue engine — demo.

Shows the three claims the engine is built on:
  1. Coupling proliferation to relaxation reproduces the real-epithelium
     (Gibson) polygon-side distribution, tunable random <-> crystalline.
  2. The relaxation step is a jitted array op — throughput vs cell count.
  3. The whole relaxation is autodiff-differentiable (inverse-design hook).

Run:  .venv/bin/python examples/demo_center_based.py
"""
from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

from jax_morpho.center_based import (
    D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT,
    grow_relax, interior_side_counts, relax,
)
from jax_morpho.stats import (
    GIBSON_EPITHELIUM_SIDES, POISSON_VORONOI_SIDES, l1_distance,
)


def _report(tag, sides):
    n = max(len(sides), 1)
    d = {k: float((sides == k).sum()) / n for k in range(4, 10)}
    fr = "  ".join(f"{k}:{d[k]:.2f}" for k in range(4, 10))
    print(f"  {tag:18s} n={len(sides):4d} mean={sides.mean():.3f} "
          f"hex={d[6]:.3f} | {fr}")
    print(f"  {'':18s} L1->PoissonVoronoi={l1_distance(d, POISSON_VORONOI_SIDES):.3f}"
          f"   L1->Gibson(real)={l1_distance(d, GIBSON_EPITHELIUM_SIDES):.3f}")


def main():
    print("1. GIBSON: proliferation<->relaxation balance "
          "(random=PV 0.29 | real ~0.45 | crystalline >0.7)")
    for relax_steps in (8, 25, 80, 250):
        pos, alive = grow_relax(n_max=1400, n_start=20, target=1000,
                                relax_steps=relax_steps, seed=1)
        _report(f"relax_steps={relax_steps}", interior_side_counts(pos, alive))

    print("\n2. SCALE: jitted relaxation-step throughput vs N (CPU float32)")
    rng = np.random.default_rng(0)
    for N in (500, 1500, 3000):
        p = jnp.asarray(rng.normal(0, np.sqrt(N) * 0.4, (N, 2)).astype(np.float32))
        al = jnp.ones(N, np.float32)
        relax(p, al, D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT, 1).block_until_ready()
        t0 = time.time()
        relax(p, al, D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT, 50).block_until_ready()
        dt = (time.time() - t0) / 50
        print(f"  N={N:5d}: {dt*1e3:6.1f} ms/step ({N/dt/1e3:6.1f}k cell-steps/s)")
    print("  (dense O(N^2); JAX-MD spatial neighbor-lists + GPU -> millions)")

    print("\n3. DIFFERENTIABLE: d(mean nearest-neighbor dist)/d(adhesion D)")
    rng = np.random.default_rng(2)
    p0 = jnp.asarray(rng.normal(0, 4, (200, 2)).astype(np.float32))
    al = jnp.ones(200, np.float32)

    def observable(D):
        p = relax(p0, al, D, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT, 200)
        diff = p[:, None, :] - p[None, :, :]
        r2 = (diff * diff).sum(-1) + jnp.eye(200) * 1e9
        return jnp.sqrt(r2.min(axis=1) + 1e-12).mean()

    print(f"  observable(D=1.0) = {float(observable(1.0)):.4f}")
    print(f"  d/dD              = {float(jax.grad(observable)(1.0)):+.4f}  "
          f"(autodiff through the whole relaxation)")


if __name__ == "__main__":
    main()
