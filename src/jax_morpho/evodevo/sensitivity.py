"""Developmental sensitivity — the three ways to compute ∂phenotype/∂parameter.

The sensitivity vector s = ∂x/∂θ is the Jacobian of the genotype→phenotype
(developmental) map — Milocco & Uller's central object (their Eq. 2), and the
thing a development-derived G-matrix is built from. There are three ways to get
it, and Phase 0b checks they agree on their toggle switch (where the answer is
known) before we trust any of them on the mechanical engine:

  forward  : forward-mode autodiff through the developmental solve
             (= integrating the variational equation ṡ = A s + b — their method)
  reverse  : reverse-mode autodiff through the solve
             (cheap for a scalar objective, but degrades through long unrolls —
              this is the failure we found on the mechanical relaxation)
  implicit : implicit-function-theorem sensitivity of the developmental
             equilibrium x* (a fixed point of ẋ = deriv(x, θ)):
                 ∂x*/∂θ = −(∂deriv/∂x)⁻¹ (∂deriv/∂θ)
             one linear solve, independent of how the equilibrium was reached —
             the correct, scalable tool.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def forward_sensitivity(develop_fn, theta):
    """∂develop_fn(theta)/∂theta via forward-mode autodiff (variational)."""
    return jax.jacfwd(develop_fn)(theta)


def reverse_sensitivity(develop_fn, theta):
    """∂develop_fn(theta)/∂theta via reverse-mode autodiff."""
    return jax.jacrev(develop_fn)(theta)


def implicit_sensitivity(deriv, x_star, theta):
    """Sensitivity of a fixed point x* of the developmental dynamics
    ``ẋ = deriv(x, theta)`` with respect to theta, by the implicit function
    theorem: ∂x*/∂θ = −(∂deriv/∂x)⁻¹ (∂deriv/∂θ).

    For an energy relaxation, ``deriv = -∇E`` and ``∂deriv/∂x = -Hessian``.
    This reference implementation forms the (small) Jacobians densely and calls
    ``jnp.linalg.solve``. For the high-dimensional mechanical engine the same IFT
    relation will be solved matrix-free (CG on Hessian-vector products); the
    interface is identical, only the linear solve changes.
    """
    A = jax.jacobian(lambda x: deriv(x, theta))(x_star)      # ∂deriv/∂x
    B = jax.jacobian(lambda th: deriv(x_star, th))(theta)    # ∂deriv/∂θ
    return -jnp.linalg.solve(A, B)
