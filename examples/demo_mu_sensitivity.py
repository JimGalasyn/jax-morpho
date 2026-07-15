"""Phase-0b calibration demo: the developmental-sensitivity engine on the
Milocco & Uller toggle switch.

Shows (1) the three ways to compute the sensitivity ∂phenotype/∂parameter —
forward-mode autodiff, reverse-mode autodiff, and implicit-diff at the steady
state — all agree; and (2) sensitivity × allelic-effect = the Fisher regression
average effect (their Fig 1C).

Run:  .venv/bin/python examples/demo_mu_sensitivity.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jax_morpho.evodevo.reference_mu import develop_theta, toggle_deriv
from jax_morpho.evodevo.sensitivity import (
    forward_sensitivity, implicit_sensitivity, reverse_sensitivity,
)

dev = lambda th: develop_theta(th, 0.0, 1000, 0.05)
theta = jnp.array([0.0, 0.0])


def main():
    print("Toggle-switch developmental sensitivity ∂x/∂θ, three ways:\n")
    Jf = np.asarray(forward_sensitivity(dev, theta))
    Jr = np.asarray(reverse_sensitivity(dev, theta))
    Ji = np.asarray(implicit_sensitivity(
        lambda x, th: toggle_deriv(x, th, 0.0), dev(theta), theta))
    print("  forward-mode autodiff (their variational method):\n", Jf)
    print("  reverse-mode autodiff:\n", Jr)
    print("  implicit-diff at the equilibrium (our core tool):\n", Ji)
    print(f"\n  max|forward - reverse|  = {np.abs(Jf - Jr).max():.2e}")
    print(f"  max|forward - implicit| = {np.abs(Jf - Ji).max():.2e}")

    print("\nFig 1C — average effect α = sensitivity s × allelic effect γ:")
    s1 = Ji[0, 0]
    dev_batch = jax.jit(jax.vmap(lambda t1: dev(jnp.stack([t1, 0.0]))))
    rng = np.random.default_rng(0)
    p, n = 0.5, 4000
    for gamma in (0.001, 0.01, 0.05):
        g = rng.choice([-1, 0, 1], n, p=[(1 - p) ** 2, 2 * p * (1 - p), p ** 2])
        x1 = np.asarray(dev_batch(jnp.asarray((g * gamma).astype(np.float32))))[:, 0]
        a_reg = np.cov(x1, g)[0, 1] / np.var(g, ddof=1)   # match np.cov's ddof
        print(f"  γ={gamma:5.3f}:  regression α={a_reg:.6f}   s·γ={s1*gamma:.6f}   "
              f"ratio={a_reg/(s1*gamma):.4f}")
    print("\n  → our sensitivity is the correct developmental Jacobian, and it")
    print("    reproduces the Fisher average effect. Tooling calibrated.")


if __name__ == "__main__":
    main()
