"""Layer C: equilibrium form → landmarks → Procrustes shape (DESIGN.md §2C).

Why landmarks. A genome does not contain a bundle of shape descriptors; it
builds a body, and morphometrics measures that body at homologous points. In the
equilibrium phase every individual develops from the same initial tissue, so
**cell index gives homology for free** and a landmark set is just a fixed subset
of cells. (Once differential growth is switched on, homology breaks and this
seam becomes the Thompson-transformation / attention engine — DESIGN.md §2C.)

Why Procrustes is load-bearing, not tidying-up
----------------------------------------------
Phase 1 measured the reason. Relaxation conserves the centre of mass exactly,
but **not orientation**: every gradient step carries zero net torque, yet net
rotation accumulates as the shape deforms — a geometric phase, the falling-cat
effect. So raw landmark coordinates are *not a function of the genome*; they
carry a path-dependent rotation. Concretely, the raw ∂x*/∂θ disagreed with
finite differences by exactly one rotational mode (~0.45 against a Jacobian of
scale ~1.3), with translations at noise.

Procrustes quotients that out, and it is what makes the genotype→phenotype map
well defined. The payoff is measurable: the *composed* ∂z/∂a matches finite
differences **with no gauge projection at all** — see
``tests/test_phenotype.py::TestGaugeInvarianceOfTheChain``.

Shape space is deliberately degenerate
--------------------------------------
Full Procrustes removes translation (2), rotation (1) and scale (1), so ``2k``
aligned coordinates carry only ``2k − 4`` degrees of freedom (:func:`shape_dim`).
The covariance of *small* variation about a mean shape — which is what G is — is
therefore **rank-deficient by exactly 4**, structurally and correctly. That is a
fact to respect, not a bug to regularise away: it is why DESIGN.md §5.2 calls for
a *non-degenerate* phenotype, and why anything downstream that wants to invert a
covariance (β = P⁻¹s) must work in the tangent space rather than call ``inv``.

The word *small* is doing real work, because the four dimensions are not lost the
same way. Three go **exactly**: the two centring conditions and Procrustes
optimality (``Σ zᵢ × refᵢ = 0``) are linear in ``z``, so their singular values
are at machine zero for any cloud. The fourth goes only **asymptotically**: unit
scale is nonlinear, shapes lie on a sphere, and at finite spread ε a cloud still
pokes out of the tangent plane by O(ε²) — leaving a singular value of relative
size O(ε) rather than zero. So the finite-ε linear rank is ``2k − 3``, and
``2k − 4`` is the ε→0 tangent dimension. Pinned in ``tests/test_phenotype.py``
(``..._three_exactly_one_asymptotically``).

Size is not lost — it is simply a separate, scalar phenotype
(:func:`centroid_size`).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def landmarks(pos, idx):
    """Landmark coordinates: a fixed subset of cells, homologous by cell index."""
    return pos[idx]


def shape_dim(n_landmarks):
    """Dimension of 2D Procrustes shape space: ``2k − 4``.

    2k coordinates, less 2 translation + 1 rotation + 1 scale. This is the
    *tangent* dimension — the rank of the covariance of small variation about a
    mean shape, and so of G. A cloud of arbitrarily different shapes has linear
    rank ``2k − 3``, because the scale constraint is nonlinear; see the module
    docstring.
    """
    return 2 * n_landmarks - 4


def centroid_size(L):
    """Centroid size: the Frobenius norm of the centred landmarks.

    The scale that Procrustes removes — kept here so it can be carried as its
    own phenotype rather than discarded.
    """
    return jnp.sqrt(((L - L.mean(0)) ** 2).sum() + 1e-30)


def _centre_and_scale(L):
    C = L - L.mean(0)
    return C / centroid_size(C)


def procrustes_align(L, ref):
    """Rotate centred+scaled landmarks ``L`` onto ``ref``. Differentiable.

    Uses the 2D closed form rather than an SVD. Minimising ``‖R(φ)L − ref‖²``
    over φ gives ``φ = atan2(Σ L×ref, Σ L·ref)`` — one ``atan2``, smooth
    everywhere except the measure-zero case of a vanishing alignment (guarded
    below). The SVD route is the textbook one for general dimension, but its
    gradient is singular when singular values coincide, which for a near-round
    tissue is not a hypothetical.
    """
    S = (L[:, 0] * ref[:, 1] - L[:, 1] * ref[:, 0]).sum()   # Σ L × ref
    C = (L * ref).sum()                                     # Σ L · ref
    phi = jnp.arctan2(S, C)
    c, s = jnp.cos(phi), jnp.sin(phi)
    R = jnp.array([[c, -s], [s, c]])
    return L @ R.T


def procrustes_shape(pos, idx, ref):
    """Equilibrium form → Procrustes shape vector ``z`` of length ``2k``.

    ``ref`` is a fixed reference configuration of the same landmarks, already
    centred and scaled (see :func:`make_reference`). Holding it fixed across
    individuals is what makes ``z`` comparable between them.
    """
    L = _centre_and_scale(landmarks(pos, idx))
    return procrustes_align(L, ref).ravel()


def make_reference(pos, idx):
    """A fixed, centred-and-scaled reference from one configuration — normally
    the equilibrium of the mean genome."""
    return _centre_and_scale(landmarks(pos, idx))


# ---------------------------------------------------------------------------
# Readout sensitivity
# ---------------------------------------------------------------------------

def shape_jacobian(pos, idx, ref):
    """∂z/∂x* — the (2k, 2N) Jacobian of the landmark+Procrustes readout.

    This is the factor that annihilates the rigid modes: an infinitesimal
    translation or rotation of the whole tissue leaves ``z`` unchanged, so this
    Jacobian kills them, and the gauge freedom left over from the implicit solve
    cannot reach the phenotype. Tested directly in
    ``TestReadoutAnnihilatesRigidModes``.
    """
    n = pos.shape[0]
    flat = lambda q: procrustes_shape(q.reshape(n, 2), idx, ref)
    return jax.jacfwd(flat)(pos.ravel())
