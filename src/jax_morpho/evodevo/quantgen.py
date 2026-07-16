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

Phase 3 completes the layer: the selection gradient ``β``, the Lande response
``Δz̄ = Gβ``, and the reverse-mode ``Jᵀβ`` path that gives the response without
ever forming G.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


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


# ---------------------------------------------------------------------------
# G from a genetic architecture (Milocco-Uller Eq. 12, generalised)
# ---------------------------------------------------------------------------

def average_effects(J, arch):
    """Fisher average effects from the developmental sensitivity: ``α_l = γ_l·s_l``.

    ``s_l`` is the sensitivity of the phenotype to the *gene* locus ``l`` writes
    to — i.e. column ``gene_of_locus[l]`` of the developmental Jacobian ``J``
    (∂phenotype/∂genome). This is Milocco & Uller's Fig-1C identity: the
    regression average effect of an allele equals its effect on the developmental
    parameter times the parameter's sensitivity. We take it from autodiff instead
    of from a regression.

    ``J``: (n_traits, n_genes). Returns ``(n_loci, n_traits)``.
    """
    J = np.asarray(J)
    return arch.gamma[:, None] * J[:, arch.gene_of_locus].T


def build_G_alleles(alpha, scores):
    """``G = Σ_l 2 p_l q_l α_l α_lᵀ`` — Milocco & Uller Eq. 12.

    ``alpha``: (n_loci, n_traits) average effects; ``scores``: (n_loci, n_ind)
    genotype scores, from which the *realised* allele frequencies are taken.
    Vectorised over loci (their MATLAB loops).
    """
    from jax_morpho.evodevo.genetics import allele_frequencies

    p, q = allele_frequencies(np.asarray(scores))
    w = 2.0 * p * q                                       # (n_loci,)
    A = np.asarray(alpha)
    return np.einsum("l,li,lj->ij", w, A, A)


# ---------------------------------------------------------------------------
# Selection and the breeder's equation
# ---------------------------------------------------------------------------

def selection_gradient(P, s, rcond=1e-10):
    """``β = P⁻¹ s`` — the selection gradient from the selection differential.

    Uses a pseudo-inverse, and the reason is structural rather than defensive: a
    covariance of *shapes* is singular in ambient coordinates by construction
    (2k coordinates, 2k−4 degrees of freedom). Work in tangent coordinates
    (:func:`jax_morpho.evodevo.phenotype.tangent_basis`) and P is genuinely
    invertible; the pinv then costs nothing and protects against the residual
    case where the population simply does not vary in some tangent direction —
    which is *not* a numerical accident but a real statement that selection on
    that direction cannot be estimated.
    """
    return np.linalg.pinv(np.asarray(P), rcond=rcond) @ np.asarray(s)


def lande_response(G, beta):
    """``Δz̄ = G β`` — the multivariate breeder's (Lande) equation."""
    return np.asarray(G) @ np.asarray(beta)


#: The reverse-mode response path ``Δz̄ = J M Jᵀβ`` lives in
#: :func:`jax_morpho.evodevo.pipeline.lande_response_vjp`, not here, because it
#: cannot be written as `jax.vjp` through the genome→phenotype map: that map ends
#: in a `lax.while_loop` (not reverse-differentiable), and unrolling it would
#: differentiate the relaxation *path* rather than the fixed point — Phase 1's
#: finding. It needs the implicit transpose
#: (:func:`jax_morpho.evodevo.fixed_point.implicit_vjp`).


def truncation_select(z, optimum, keep_fraction=0.5):
    """Truncation selection: keep the ``keep_fraction`` closest to ``optimum``.

    Returns ``(selected_indices, selection_differential)`` where the differential
    ``s = z̄_selected − z̄_all`` is the within-generation shift selection produces
    — *before* any inheritance. Milocco & Uller's protocol.
    """
    z = np.asarray(z)
    d = np.sqrt(((z - np.asarray(optimum)) ** 2).sum(1))
    n_keep = int(keep_fraction * z.shape[0])
    sel = np.argsort(d)[:n_keep]
    return sel, z[sel].mean(0) - z.mean(0)


def angle_deg(u, v):
    """Angle between two response vectors, in degrees — the Fig-3C metric.

    Fig 3C compares *directions*: a G that predicts the response points the same
    way as the realised response, while P misaligns at low allele frequency.
    """
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
