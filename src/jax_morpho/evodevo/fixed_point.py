"""Implicit sensitivity of a developmental equilibrium — the Phase 1 engine.

Development is written as a fixed point ``F(x*, θ) = 0``. By the implicit
function theorem the developmental sensitivity is

    ∂x*/∂θ = −[∂F/∂x]⁻¹ [∂F/∂θ]

costing **one linear solve per parameter**, independent of how many relaxation
steps were taken to reach x*. Energy minimisation (``F = −∇E``) is the default
instance, but the engine is written against the general fixed point so it also
covers non-potential and growth-driven development (docs/DESIGN.md §1).

The gauge problem
-----------------
For a mechanical tissue this inverse **does not exist**. The energy depends only
on pairwise distances, so it is invariant under rigid motions of the whole
tissue, and at a true equilibrium ``∂F/∂x`` is exactly singular with a
3-dimensional null space in 2D: two translations and one rotation. (Measured on
a converged 12-cell blob: eigenvalues ``[0, 0, 0, 3.19, 6.00, ...]``.) A plain
``jnp.linalg.solve`` is not merely inaccurate here — the system is degenerate.

The resolution is the **pseudo-inverse**: the minimum-norm solution, which is
exactly the one with no rigid-motion component. Because E is rigid-invariant for
every θ, ``∂F/∂θ`` is orthogonal to the null space too, so the system is
consistent and the pseudo-inverse solves it exactly rather than in a
least-squares sense.

That choice is not free, and the difference is measurable (see
``tests/test_fixed_point.py::TestGaugeIsRotationOnly``). A pairwise
central-force gradient step carries zero net force and zero net torque, so each
step is orthogonal to the *current* rigid modes. Integrating that along a path
gives two different answers:

* **Translation** is genuinely conserved: the centre of mass of x*(θ) is that of
  the initial condition, to 1e-15, for every θ. The pseudo-inverse agrees.
* **Rotation is not.** Zero torque at every instant does *not* integrate to zero
  net rotation, because the modes rotate with the shape as it changes — a
  geometric phase, the same anholonomy that lets a falling cat land on its feet
  with zero angular momentum throughout. Finite-differencing the relaxation
  therefore reports a large rotational component (~0.45 against a Jacobian of
  scale ~1.3) that no fixed-point method can reproduce, because it is not a
  property of the fixed point at all. Its *size* depends on the path: changing
  the solver's step schedule moved it from ~5.2 to ~0.45 while leaving the
  gauge-invariant sensitivity untouched, which is the clearest statement of what
  it is.

So the equilibrium **form** is a function of θ, while its **orientation** is a
functional of the entire developmental trajectory. This is why the phenotype
readout is Procrustes-aligned landmarks (DESIGN.md §2C): the alignment is not
tidying-up before comparison, it is what makes the phenotype a well-defined
function of the genotype in the first place. The gauge is a property of the
representation, not of the form.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Rigid-motion null space
# ---------------------------------------------------------------------------

def rigid_modes(pos, alive=None):
    """Orthonormal basis of the rigid-motion null space of a 2D tissue.

    Returns a ``(2N, 3)`` matrix whose columns span {x-translation,
    y-translation, rotation about the centroid}, restricted to alive cells and
    orthonormalised. These are the exact zero modes of the Hessian of any
    rigid-invariant energy at equilibrium.

    Dead (padded) cells contribute their own trivial zero modes — their forces
    vanish identically, so the Jacobian has zero rows and columns there — but
    those are **not** in this basis.

    .. warning::
       That omission is only safe for the ``pinv`` path, which discovers every
       null direction itself. **The CG paths do not**: handing this basis to
       ``fixed_point_sensitivity(solver='cg')``, :func:`implicit_vjp` or
       :func:`implicit_jvp` on a *padded* tissue without also passing ``alive``
       leaves the dead-cell directions unhandled, and CG wanders into them and
       returns **NaN** (measured on a padded tissue, against a dense reference
       that returns a finite answer; found by Copilot on PR #3). Pass ``alive``
       to those functions — :func:`energy_sensitivity` and the ``pipeline``
       helpers do it for you — or use ``pinv``.
    """
    pos = jnp.asarray(pos, float)
    n = pos.shape[0]
    m = jnp.ones(n) if alive is None else jnp.asarray(alive, float)

    c = (pos * m[:, None]).sum(0) / jnp.maximum(m.sum(), 1e-12)
    rel = (pos - c) * m[:, None]

    tx = jnp.stack([m, jnp.zeros(n)], -1).ravel()
    ty = jnp.stack([jnp.zeros(n), m], -1).ravel()
    rot = jnp.stack([-rel[:, 1], rel[:, 0]], -1).ravel()

    Z = jnp.stack([tx, ty, rot], -1)
    Q, _ = jnp.linalg.qr(Z)
    return Q


def project_out(Z, v):
    """Remove the components of ``v`` lying in the span of orthonormal ``Z``."""
    return v - Z @ (Z.T @ v)


def _flat_mask(alive, n_flat):
    """(N,) alive mask → (2N,) mask over interleaved [x, y] coordinates."""
    if alive is None:
        return jnp.ones(n_flat)
    return jnp.repeat(jnp.asarray(alive, float), 2)


def _projected_operator(Av, Z, mask):
    """The nonsingular stand-in for a singular ``A``, used by every CG path here.

    ``A`` is singular in two independent ways and both must be neutralised, or CG
    wanders off into the null space and returns NaN:

    * the **rigid modes**, spanned by ``Z`` — always present for a
      rigid-invariant energy;
    * one trivial mode per **dead/padded cell** — their forces vanish
      identically, so ``A`` has zero rows and columns there. ``mask`` handles
      these. Passing ``alive=None`` (no padding) makes this term vanish.

    Both are removed the same way: make the operator the **identity** along them.
    On the physical subspace it is still ``P(−A)P``, which is PSD at a minimum,
    so the composite is symmetric positive-definite and CG is entitled to run::

        M = P (−A) P  +  Z Zᵀ  +  (I − mask)

    Factored out because it was written three times and the dead-cell term was
    present in only one of them — which is exactly how
    ``implicit_vjp`` came to return NaN on a padded tissue (found by Copilot on
    PR #3, confirmed against the dense reference).
    """
    def M(v):
        w = project_out(Z, v * mask)
        Aw = Av(w) * mask
        return (project_out(Z, -Aw)          # physical subspace: P(−A)P
                + Z @ (Z.T @ v)              # identity on the rigid modes
                + v * (1.0 - mask))          # identity on dead cells
    return M


# ---------------------------------------------------------------------------
# Implicit sensitivity
# ---------------------------------------------------------------------------

def fixed_point_sensitivity(F, x_star, theta, *, solver="pinv",
                            null_basis=None, alive=None, rcond=1e-8,
                            cg_tol=1e-10):
    """∂x*/∂θ for a fixed point ``F(x*, θ) = 0``.

    Parameters
    ----------
    F : callable ``F(x, theta) -> array`` shaped like ``x``
        The developmental residual. For energy relaxation, ``F = −∇E``.
    x_star : array
        A converged fixed point. **Must** satisfy ``F(x*, θ) ≈ 0``; this is not
        checked here (see :func:`~jax_morpho.evodevo.mechanical.equilibrate`,
        which reports its residual).
    theta : array, shape (m,)
    solver : {"pinv", "cg"}
        ``pinv`` forms ``∂F/∂x`` densely and applies the pseudo-inverse: exact,
        gauge-safe, O(n³) — the reference path and the one gate #1 validates.
        ``cg`` is matrix-free (Hessian-vector products, no matrix formed) and
        projects the rigid modes out of the Krylov space; it requires ``∂F/∂x``
        to be symmetric negative-semidefinite, i.e. an energy minimum. This is
        the path that scales.
    null_basis : array, shape (n, k), optional
        Orthonormal basis of known exact zero modes (see :func:`rigid_modes`).
        Required by ``cg``; optional for ``pinv``, which discovers the null
        space itself via ``rcond``.

    Returns
    -------
    J : array, shape (n, m) — the developmental Jacobian ∂x*/∂θ, flattened over
    x's leading axes.
    """
    shape = x_star.shape
    flat = lambda v: v.reshape(-1)
    Ffun = lambda xf, th: flat(F(xf.reshape(shape), th))
    xf = flat(x_star)

    # B = ∂F/∂θ  (n, m). Orthogonal to the rigid modes by rigid-invariance of E.
    B = jax.jacobian(lambda th: Ffun(xf, th))(theta)

    if solver == "pinv":
        A = jax.jacobian(lambda x: Ffun(x, theta))(xf)          # (n, n)
        return -jnp.linalg.pinv(A, rcond=rcond) @ B

    if solver == "cg":
        if null_basis is None:
            raise ValueError("solver='cg' requires null_basis (see rigid_modes)")
        Z = null_basis
        mask = _flat_mask(alive, xf.size)

        # Matrix-free A·v via a JVP through F — no Hessian is ever formed.
        def Av(v):
            return jax.jvp(lambda x: Ffun(x, theta), (xf,), (v,))[1]

        M = _projected_operator(Av, Z, mask)

        def solve_col(b):
            x, _ = jax.scipy.sparse.linalg.cg(M, project_out(Z, b * mask),
                                              tol=cg_tol, atol=0.0,
                                              maxiter=10 * xf.size)
            return x

        return jax.vmap(solve_col, in_axes=1, out_axes=1)(B)

    raise ValueError(f"unknown solver {solver!r}; expected 'pinv' or 'cg'")


def implicit_vjp(F, x_star, theta, v, *, null_basis=None, alive=None,
                 cg_tol=1e-10, cg_maxiter=400):
    """``(∂x*/∂θ)ᵀ v`` — the transpose sensitivity, in **one linear solve**.

    This is the path DESIGN.md §2D needs and the reason `Δz̄ = Jᵀβ`-style
    quantities are affordable at scale. Forming ``∂x*/∂θ`` costs one solve *per
    parameter*; this costs one solve **total**, whatever the parameter count.

    Why it cannot be `jax.vjp` through the solver
    ---------------------------------------------
    Two independent reasons, and they are worth stating because the naive route
    looks so available:

    1. **It does not run.** `equilibrate` is a `lax.while_loop` (it iterates to a
       tolerance, not a fixed count), and reverse-mode autodiff through
       `while_loop` is not defined in JAX. Rewriting it as a `scan` would make it
       run, which brings us to:
    2. **It would be the wrong answer.** That is Phase 1's whole finding —
       backprop through an unrolled relaxation differentiates the *path*, not the
       fixed point, and silently returns garbage if the relaxation has not
       genuinely converged (see §1 of docs/DESIGN.md, and the period-2 limit
       cycle that started all this).

    The implicit transpose has neither problem: it never touches the iteration.
    From ``∂x*/∂θ = −A⁻¹B`` with ``A = ∂F/∂x`` and ``B = ∂F/∂θ``,

        (∂x*/∂θ)ᵀ v = −Bᵀ A⁻ᵀ v

    so: solve ``A w = v`` (one projected CG, exploiting ``A = Aᵀ`` for an energy),
    then apply ``Bᵀ`` — a plain VJP through an *explicit* function, which
    reverse-mode handles fine because there is no solve inside it.
    """
    shape = x_star.shape
    Ffun = lambda xf, th: F(xf.reshape(shape), th).reshape(-1)
    xf = x_star.reshape(-1)
    vf = jnp.asarray(v).reshape(-1)

    if null_basis is None:
        raise ValueError("implicit_vjp requires null_basis (see rigid_modes)")
    Z = null_basis
    mask = _flat_mask(alive, xf.size)

    def Av(w):
        return jax.jvp(lambda x: Ffun(x, theta), (xf,), (w,))[1]

    M = _projected_operator(Av, Z, mask)
    w, _ = jax.scipy.sparse.linalg.cg(M, project_out(Z, vf * mask), tol=cg_tol,
                                      atol=0.0, maxiter=cg_maxiter)
    # solve gave (−A)⁻¹v, so −A⁻¹v = +w; then (∂x*/∂θ)ᵀ v = −Bᵀ(A⁻¹v) = Bᵀ w
    _, vjp_theta = jax.vjp(lambda th: Ffun(xf, th), theta)
    return vjp_theta(w)[0]


def implicit_jvp(F, x_star, theta, dtheta, *, null_basis=None, alive=None,
                 cg_tol=1e-10, cg_maxiter=400):
    """``(∂x*/∂θ) δθ`` — the forward twin of :func:`implicit_vjp`, one solve.

    ``= −A⁻¹ B δθ``: push ``δθ`` through ``B = ∂F/∂θ`` (an explicit JVP), then one
    projected CG. Together the two directions make ``J M Jᵀ β`` cost **two**
    solves instead of one per parameter.
    """
    shape = x_star.shape
    Ffun = lambda xf, th: F(xf.reshape(shape), th).reshape(-1)
    xf = x_star.reshape(-1)

    if null_basis is None:
        raise ValueError("implicit_jvp requires null_basis (see rigid_modes)")
    Z = null_basis
    mask = _flat_mask(alive, xf.size)

    def Av(w):
        return jax.jvp(lambda x: Ffun(x, theta), (xf,), (w,))[1]

    M = _projected_operator(Av, Z, mask)
    B_dtheta = jax.jvp(lambda th: Ffun(xf, th), (theta,), (dtheta,))[1]
    # solve (−A) w = B δθ  ⇒  w = −A⁻¹ B δθ = (∂x*/∂θ) δθ
    w, _ = jax.scipy.sparse.linalg.cg(M, project_out(Z, B_dtheta * mask),
                                      tol=cg_tol, atol=0.0, maxiter=cg_maxiter)
    return w.reshape(shape)


def energy_sensitivity(E, x_star, alive, theta, **kw):
    """∂x*/∂θ for an energy minimum — the ``F = −∇E`` instance of
    :func:`fixed_point_sensitivity`, with the rigid modes supplied.

    ``E`` is called as ``E(pos, alive, theta)``.
    """
    F = lambda x, th: -jax.grad(E)(x, alive, th) * alive[:, None]
    kw.setdefault("null_basis", rigid_modes(x_star, alive))
    kw.setdefault("alive", alive)      # padded cells are null directions too
    return fixed_point_sensitivity(F, x_star, theta, **kw)


# ---------------------------------------------------------------------------
# Gate #1 reference: finite differences
# ---------------------------------------------------------------------------

def finite_difference_sensitivity(develop_fn, theta, eps=1e-5):
    """∂develop_fn(θ)/∂θ by central differences — the known-answer reference the
    implicit sensitivity is gated against (docs/DESIGN.md validation ladder #1).

    Deliberately naive and slow: it re-runs development ``2m`` times and knows
    nothing about the implicit machinery, which is exactly what makes it an
    independent check.
    """
    theta = jnp.asarray(theta, float)
    cols = []
    for k in range(theta.shape[0]):
        e = jnp.zeros_like(theta).at[k].set(eps)
        xp = jnp.asarray(develop_fn(theta + e)).reshape(-1)
        xm = jnp.asarray(develop_fn(theta - e)).reshape(-1)
        cols.append((xp - xm) / (2 * eps))
    return jnp.stack(cols, -1)
