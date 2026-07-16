"""Layer D (opening move): the additive-genetic covariance from a developmental
Jacobian — ``G = J M Jᵀ`` (docs/DESIGN.md §2D).

This is the delta method: push the genetic covariance ``M`` through a local
linearisation of the genotype→phenotype map. It is exactly Milocco & Uller's
Eq. 12, ``G = Σ σᵢ² sᵢ sᵢᵀ``, written in matrix form — their ``sᵢ`` are the
columns of ``J``.

The claim being made is narrow and testable, which is the point: *if* the
developmental map were linear over the spread of the population, then ``J M Jᵀ``
would be the phenotypic covariance exactly. It is not linear, so ``G`` is a
first-order approximation whose error must vanish with the perturbation scale —
gate #2 (``tests/test_quantgen.py``) measures that it does, and at the predicted
first order.

Phase 2 stops here, at the object and its gate. The breeder's equation
(``β``, ``Δz̄ = Gβ``, and the reverse-mode ``Jᵀβ`` path that skips forming G) is
Phase 3.
"""
from __future__ import annotations

import jax.numpy as jnp


def build_G(J, M):
    """``G = J M Jᵀ`` — genetic covariance ``M`` pushed through the
    developmental Jacobian ``J`` (∂phenotype/∂genome).

    Note ``G`` inherits ``J``'s rank. With a Procrustes shape phenotype, ``J``
    annihilates the 4 rigid-and-scale directions, so ``G`` is rank-deficient by
    exactly 4 — structurally, not numerically. See
    :mod:`jax_morpho.evodevo.phenotype`.
    """
    return J @ M @ J.T


def empirical_covariance(samples):
    """Unbiased covariance of stacked samples, shape (n_samples, dim)."""
    X = samples - samples.mean(0)
    return (X.T @ X) / (X.shape[0] - 1)


def relative_difference(A, B):
    """Frobenius relative difference ‖A − B‖ / ‖B‖ — the gate metric."""
    return float(jnp.linalg.norm(A - B) / (jnp.linalg.norm(B) + 1e-300))
