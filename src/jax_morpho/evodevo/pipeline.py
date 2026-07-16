"""The composed developmental pipeline: genome → θ → x* → landmarks → shape.

This is the spine of docs/DESIGN.md §2, wired end to end::

    a  ──[A: GRN/MLP]──▶  θ  ──[B: relax to x*]──▶  x*  ──[C: Procrustes]──▶  z
         genome_map            mechanical               phenotype

and its Jacobian, assembled by the chain rule from the three layers:

    ∂z/∂a  =  (∂z/∂x*) · (∂x*/∂θ) · (∂θ/∂a)
              readout     implicit      autodiff

Only the middle factor needs the implicit function theorem — it is the only one
with a solve inside it. The outer two are explicit functions, so plain autodiff
is not merely adequate there, it is exact.

The two load-bearing interfaces are θ and z (DESIGN.md §2), which is why they are
plain arrays: every layer here is swappable without touching its neighbours.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_morpho.evodevo.fixed_point import (
    energy_sensitivity, implicit_jvp, implicit_vjp, rigid_modes,
)
from jax_morpho.evodevo.genome_map import GRN, grn_field, grn_jacobian
from jax_morpho.evodevo.mechanical import (
    equilibrate, field_morse_energy, hex_blob,
)
from jax_morpho.evodevo.phenotype import (
    make_reference, procrustes_shape, shape_jacobian,
)


class Organism(NamedTuple):
    """Everything fixed across individuals: the conserved developmental map,
    the initial tissue, the landmark scheme, and the Procrustes reference.

    A *population* varies only in the genome ``a`` fed through this.
    """
    pos0: jnp.ndarray       # (N,2) initial tissue — must be cohesive
    alive: jnp.ndarray      # (N,)
    coords: jnp.ndarray     # (N,2) positional information read by the GRN
    grn: GRN                # the fixed, conserved genome→θ map
    idx: jnp.ndarray        # (k,) landmark cell indices — homology by index
    ref: jnp.ndarray        # (k,2) Procrustes reference (centred + scaled)


def theta_of(a, org: Organism):
    """Layer A: genome → per-cell developmental parameter field."""
    return grn_field(a, org.coords, org.grn)


def develop(a, org: Organism, tol=1e-12):
    """Layers A+B: genome → equilibrium form x*.

    Raises if development failed to reach equilibrium, rather than passing a
    non-fixed-point downstream where the sensitivity would be meaningless.
    """
    x, res, _, ok = equilibrate(org.pos0, org.alive, theta_of(a, org), tol=tol)
    if not bool(ok):
        raise RuntimeError(
            f"development did not reach equilibrium: max|F| = {float(res):.3e}")
    return x


def phenotype(a, org: Organism, tol=1e-12):
    """The full map: genome → Procrustes shape ``z`` (2k,)."""
    return procrustes_shape(develop(a, org, tol), org.idx, org.ref)


def develop_population(A, org: Organism, tol=1e-12):
    """Develop a whole population at once: genomes ``A`` (n, n_genes) → forms
    ``(n, N, 2)``, plus per-individual residual and convergence flag.

    ``vmap`` over individuals — the GPU-parallel path DESIGN.md §2E calls for,
    and the reason :func:`develop`'s convergence check is *not* used here: a
    Python ``bool()`` on a traced value would force a host sync per individual
    and forbid batching. The flags come back as an array so the caller checks
    the population in one shot (see :func:`phenotype_population`).
    """
    def one(a):
        x, res, _, ok = equilibrate(org.pos0, org.alive, theta_of(a, org),
                                    tol=tol)
        return x, res, ok

    return jax.vmap(one)(A)


def phenotype_population(A, org: Organism, tol=1e-12):
    """Genomes ``A`` (n, n_genes) → shapes ``Z`` (n, 2k), developed in one
    batched pass. Raises if *any* individual failed to equilibrate."""
    X, res, ok = develop_population(A, org, tol=tol)
    if not bool(jnp.all(ok)):
        bad = int((~ok).sum())
        raise RuntimeError(
            f"{bad}/{A.shape[0]} individuals failed to reach equilibrium "
            f"(worst max|F| = {float(res.max()):.3e})")
    return jax.vmap(lambda x: procrustes_shape(x, org.idx, org.ref))(X)


def phenotype_jacobian(a, org: Organism, solver="pinv", tol=1e-12):
    """∂z/∂a — the developmental sensitivity of shape to the genome.

    The chain rule across the three layers, with the implicit function theorem
    supplying the middle factor. Gauge-invariant by construction: the readout
    factor annihilates the rigid modes, so whatever gauge the implicit solve
    picked cannot reach ``z``.
    """
    theta = theta_of(a, org)
    x = develop(a, org, tol)
    dz_dx = shape_jacobian(x, org.idx, org.ref)                       # (2k, 2N)
    dx_dtheta = energy_sensitivity(field_morse_energy, x, org.alive,
                                   theta, solver=solver)              # (2N, 2N)
    dtheta_da = grn_jacobian(a, org.coords, org.grn)                  # (2N, n_genes)
    return dz_dx @ dx_dtheta @ dtheta_da


def phenotype_vjp(a, org: Organism, beta, solver_kw=None, tol=1e-12):
    """``Jᵀβ`` for the composed chain — **one linear solve**, any number of genes.

    Right-to-left through `∂z/∂a = (∂z/∂x*)(∂x*/∂θ)(∂θ/∂a)`:

    * ``(∂z/∂x*)ᵀβ`` — VJP of the readout: explicit, plain reverse-mode.
    * ``(∂x*/∂θ)ᵀ·`` — the **implicit** transpose: one CG solve
      (:func:`~jax_morpho.evodevo.fixed_point.implicit_vjp`).
    * ``(∂θ/∂a)ᵀ·`` — VJP of the GRN: explicit.

    Note the middle step is *not* `jax.vjp` through `equilibrate`: that neither
    runs (it is a `lax.while_loop`) nor would be correct if it did (Phase 1). The
    implicit transpose is what makes a reverse-mode path exist at all here.
    """
    kw = dict(solver_kw or {})
    theta = theta_of(a, org)
    x = develop(a, org, tol)

    # readout: (∂z/∂x*)ᵀ β
    _, vjp_read = jax.vjp(lambda q: procrustes_shape(q, org.idx, org.ref), x)
    v_x = vjp_read(jnp.asarray(beta))[0]

    # equilibrium: (∂x*/∂θ)ᵀ v_x — the one solve
    F = lambda q, th: -jax.grad(field_morse_energy)(q, org.alive, th) * org.alive[:, None]
    kw.setdefault("null_basis", rigid_modes(x, org.alive))
    kw.setdefault("alive", org.alive)      # padded cells are null directions too
    v_theta = implicit_vjp(F, x, theta, v_x, **kw)

    # genome map: (∂θ/∂a)ᵀ v_theta
    _, vjp_grn = jax.vjp(lambda g: grn_field(g, org.coords, org.grn), a)
    return vjp_grn(v_theta)[0]


def phenotype_jvp(a, org: Organism, da, solver_kw=None, tol=1e-12):
    """``J δa`` for the composed chain — one linear solve. Forward twin of
    :func:`phenotype_vjp`."""
    kw = dict(solver_kw or {})
    theta = theta_of(a, org)
    x = develop(a, org, tol)

    dtheta = jax.jvp(lambda g: grn_field(g, org.coords, org.grn), (a,), (da,))[1]
    F = lambda q, th: -jax.grad(field_morse_energy)(q, org.alive, th) * org.alive[:, None]
    kw.setdefault("null_basis", rigid_modes(x, org.alive))
    kw.setdefault("alive", org.alive)      # padded cells are null directions too
    dx = implicit_jvp(F, x, theta, dtheta, **kw)
    return jax.jvp(lambda q: procrustes_shape(q, org.idx, org.ref), (x,), (dx,))[1]


def lande_response_vjp(a, org: Organism, beta, M, tol=1e-12):
    """``Δz̄ = J M Jᵀ β`` **without ever forming G or J** — two solves total.

    The reverse-mode path DESIGN.md §2D calls for. Forming ``J`` costs one solve
    per gene; this costs one solve each way, independent of both the gene count
    and the trait count. At organism scale — where a solve is a matrix-free CG
    over ~10⁶ unknowns — that is the difference between tractable and not.
    """
    JT_beta = phenotype_vjp(a, org, beta, tol=tol)
    return phenotype_jvp(a, org, jnp.asarray(M) @ JT_beta, tol=tol)


def make_organism(grn: GRN, a_ref, n_rings=2, landmark_stride=1, jitter=0.02,
                  seed=0):
    """Build an organism: relax the mean genome, and use *its* equilibrium as the
    Procrustes reference.

    The reference has to come from somewhere and be held fixed; the mean
    genome's own form is the natural choice, and it keeps every individual's
    alignment a small rotation away from the reference rather than an arbitrary
    one.
    """
    P = hex_blob(n_rings)
    n = P.shape[0]
    key = jax.random.key(seed)
    pos0 = P + jax.random.normal(key, (n, 2)) * jitter
    alive = jnp.ones(n)
    idx = jnp.arange(0, n, landmark_stride)

    stub = Organism(pos0=pos0, alive=alive, coords=pos0, grn=grn, idx=idx,
                    ref=jnp.zeros((idx.shape[0], 2)))
    x_ref = develop(a_ref, stub)
    return stub._replace(ref=make_reference(x_ref, idx))
