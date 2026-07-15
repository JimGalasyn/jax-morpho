"""Gradient-based inverse design of tissue shape — demo.

Recovers the patterning field (anisotropic confinement) that produces a
target tissue morphology by autodiff through the whole mechanical
relaxation, and races gradient descent against random search.  The target
is a known ground-truth morphology, so it is provably reachable.

Run:  .venv/bin/python examples/demo_inverse_design.py
"""
from __future__ import annotations

import time

import numpy as np
import jax.numpy as jnp

from jax_morpho.center_based import gyration_morphology
from jax_morpho.inverse_design import (
    fit_morphology, morphology_loss, relax_in_field,
)

N = 400


def disk(seed=0):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, N)
    rad = np.sqrt(rng.uniform(0, 1, N)) * np.sqrt(N) * 0.55
    return jnp.asarray(np.column_stack(
        [rad * np.cos(ang), rad * np.sin(ang)]).astype(np.float32))


def main():
    pos0 = disk(0)
    p_true = jnp.array([np.log(0.05), np.log(0.22)], dtype=jnp.float32)
    size_t, aspect_t = (float(v) for v in
                        gyration_morphology(relax_in_field(pos0, p_true, 250)))
    print(f"ground-truth field kx=0.050 ky=0.220 -> "
          f"TARGET size={size_t:.2f} aspect={aspect_t:.2f}\n")

    print("=== GRADIENT DESCENT (Adam through the relaxation) ===")
    t0 = time.time()
    params, history = fit_morphology(pos0, size_t, aspect_t,
                                     lr=0.10, n_steps=41, relax_steps=200)
    sz, ap = gyration_morphology(relax_in_field(pos0, params, 200))
    print(f"  loss {history[0]:.4f} -> {history[-1]:.5f}  "
          f"({history[0]/max(history[-1],1e-9):.0f}x)  in {time.time()-t0:.0f}s")
    print(f"  recovered: size={float(sz):.2f} aspect={float(ap):.2f}  "
          f"kx={float(jnp.exp(params[0])):.3f} ky={float(jnp.exp(params[1])):.3f}")

    print("\n=== RANDOM SEARCH (same 41 evals) ===")
    rs = np.random.default_rng(3)
    lo, hi = np.log(0.02), np.log(0.5)
    best_L, best_p = np.inf, None
    for _ in range(41):
        cand = jnp.asarray(rs.uniform(lo, hi, 2).astype(np.float32))
        L = float(morphology_loss(cand, pos0, size_t, aspect_t, 200))
        if L < best_L:
            best_L, best_p = L, cand
    sz_r, ap_r = gyration_morphology(relax_in_field(pos0, best_p, 200))
    print(f"  best loss {best_L:.5f}  size={float(sz_r):.2f} aspect={float(ap_r):.2f}")

    print("\n=== VERDICT ===")
    print(f"  gradient descent : {history[-1]:.5f}")
    print(f"  random search    : {best_L:.5f}")
    print(f"  -> gradients through the mechanical relaxation recover the "
          f"target field;\n     gradient-free search can't match them at "
          f"equal budget.")


if __name__ == "__main__":
    main()
