"""Mechanical development: a per-cell parameter field θ → equilibrium form x*.

This is Phase 1's instance of the developmental map. Where the Milocco-Uller
reference model (``reference_mu``) integrates a 2-gene ODE to t=50, here
development is **relaxation of a tissue to mechanical equilibrium**, and the
developmental parameters are a *per-cell field* rather than two scalars.

θ is a flat vector packing two per-cell fields:

    D[i]      adhesion well depth of cell i   (must be > 0)
    r_eq[i]   preferred spacing of cell i     (must be > 0)

combined pairwise by the standard mixing rules ``D_ij = sqrt(D_i D_j)`` and
``r_eq_ij = (r_eq_i + r_eq_j)/2``, so a uniform field reproduces the global-
parameter Morse potential of :mod:`jax_morpho.center_based`. Differential
adhesion — cells of different types sticking to their own kind — is the
load-bearing sorting mechanism, so making D a per-cell field is what gives the
genome something worth writing into (layer A of docs/DESIGN.md).

Two deliberate departures from ``center_based``, both required by implicit
differentiation:

1. **A C² cutoff.** ``center_based.morse_energy`` buys continuity at ``r_max``
   by subtracting a constant from every pair, but its derivative still jumps
   there — the force is discontinuous and the Hessian carries a delta. Here a
   quintic switching function with vanishing first *and second* derivatives at
   both ends takes the potential smoothly to zero over ``[r_on, r_max]``.

   The two energies are therefore not equal even for a uniform field: they
   differ by that constant offset per interacting pair. What agrees — exactly,
   for pairs inside ``r_on`` — is the **forces**, which is what determines the
   equilibrium, since a constant has no gradient. The offset is precisely what
   ``center_based`` pays to paper over a cutoff this module handles properly.

2. **A solver that actually converges** (:func:`equilibrate`), reporting its
   residual. See that function's docstring: the fixed-step relaxation in
   ``center_based`` does *not* reach a fixed point, and implicit differentiation
   is meaningless without one.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from jax_morpho.evodevo.fixed_point import (
    _projected_operator, project_out, rigid_modes,
)

# Steepness and interaction cutoff stay global scalars in Phase 1; only the
# adhesion and spacing fields are per-cell (i.e. genetically addressable).
A_DEFAULT = 2.5
R_MAX_DEFAULT = 1.8
R_ON_FRAC = 0.8          # switching shell starts at R_ON_FRAC * r_max


# ---------------------------------------------------------------------------
# θ packing
# ---------------------------------------------------------------------------

def pack_theta(D, r_eq):
    """Pack per-cell adhesion and spacing fields into a flat θ vector."""
    return jnp.concatenate([jnp.asarray(D, float).ravel(),
                            jnp.asarray(r_eq, float).ravel()])


def unpack_theta(theta):
    """Split a flat θ vector back into (D, r_eq) per-cell fields."""
    n = theta.shape[0] // 2
    return theta[:n], theta[n:]


def uniform_theta(n, D=1.0, r_eq=1.0):
    """A spatially uniform θ field — the ``center_based`` default, as a field."""
    return pack_theta(jnp.full((n,), float(D)), jnp.full((n,), float(r_eq)))


# ---------------------------------------------------------------------------
# Initial tissue
# ---------------------------------------------------------------------------

def hex_blob(n_rings, spacing=1.0):
    """A compact hexagonal blob of ``1 + 3R(R+1)`` cells in ``n_rings`` rings.

    The initial condition matters more than it looks. A Gaussian cloud of cells
    is not a tissue — with cells scattered on the scale of ``r_max`` it relaxes
    into *disconnected fragments*, and then the equilibrium is not isolated even
    modulo a global rigid motion: each fragment carries its own zero modes, the
    fragments' relative placement costs no energy, and ∂x*/∂θ does not exist.
    (Symptom: the Hessian shows more than three zero modes, and the sensitivity
    blows up as finite differences invert a near-null direction.)

    A blob at ``spacing ≈ r_eq`` keeps every cell in contact with its neighbours,
    which is both the biologically sensible tissue and the regime in which the
    implicit function theorem applies.
    """
    pts = []
    for q in range(-n_rings, n_rings + 1):
        for r in range(max(-n_rings, -q - n_rings),
                       min(n_rings, -q + n_rings) + 1):
            pts.append((spacing * (q + 0.5 * r),
                        spacing * (jnp.sqrt(3.0) / 2.0) * r))
    return jnp.asarray(pts, float)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def _switch(r, r_on, r_max):
    """Quintic switch: 1 below r_on, 0 above r_max, with S' = S'' = 0 at both
    ends. Makes the truncated potential C² so the Hessian is well defined."""
    x = jnp.clip((r - r_on) / (r_max - r_on), 0.0, 1.0)
    return 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)


@partial(jax.jit, static_argnames=("a", "r_max"))
def field_morse_energy(pos, alive, theta, a=A_DEFAULT, r_max=R_MAX_DEFAULT):
    """Total Morse pair energy with per-cell adhesion/spacing fields.

    pos: (N,2); alive: (N,) in {0,1}; theta: (2N,) packed by :func:`pack_theta`.
    Dead cells and pairs beyond ``r_max`` contribute nothing.
    """
    D, r_eq = unpack_theta(theta)
    diff = pos[:, None, :] - pos[None, :, :]
    r = jnp.sqrt((diff * diff).sum(-1) + 1e-12)

    D_ij = jnp.sqrt(D[:, None] * D[None, :])
    r_eq_ij = 0.5 * (r_eq[:, None] + r_eq[None, :])

    morse = D_ij * ((1.0 - jnp.exp(-a * (r - r_eq_ij))) ** 2 - 1.0)
    u = morse * _switch(r, R_ON_FRAC * r_max, r_max)

    n = pos.shape[0]
    pair = alive[:, None] * alive[None, :] * (1.0 - jnp.eye(n))
    return 0.5 * (u * pair).sum()


def force_residual(pos, alive, theta, a=A_DEFAULT, r_max=R_MAX_DEFAULT):
    """F(x, θ) = −∇ₓE, masked to alive cells.

    This is the fixed-point residual of the developmental dynamics: ``F = 0``
    defines the equilibrium form x*. It is the function handed to
    :func:`jax_morpho.evodevo.fixed_point.fixed_point_sensitivity`.
    """
    g = jax.grad(field_morse_energy)(pos, alive, theta, a, r_max)
    return -g * alive[:, None]


# ---------------------------------------------------------------------------
# Equilibration
# ---------------------------------------------------------------------------

class EquilibrateState(NamedTuple):
    """Everything needed to continue a relaxation, and nothing else.

    ``lr`` is the field that is easy to omit and expensive to omit. The descent
    stage carries an adaptive rate (``min(step*2, 1.0)`` between iterations), so
    a resumed run that restarts the line search at ``lr0`` does not crash — it
    silently takes a different trajectory. That is drift, and drift is what a
    checkpoint exists to prevent. Same reason ``jax_solitons.runfns``
    checkpoints the Adam moments and not just the field.

    ``phase`` is 0 = Armijo descent, 1 = damped Newton, 2 = done. It is carried
    state rather than a recomputed predicate, so a chunk boundary landing on the
    handoff resumes on the side the monolithic loop would have been on.
    """
    pos: jnp.ndarray      # (N, 2) current positions
    lr: jnp.ndarray       # ()     adaptive descent step, carried between iters
    res: jnp.ndarray      # ()     max|F|
    n_gd: jnp.ndarray     # ()     descent iterations consumed
    n_nt: jnp.ndarray     # ()     Newton iterations consumed
    phase: jnp.ndarray    # ()     0 descent, 1 newton, 2 done

    @property
    def n_iter(self):
        return self.n_gd + self.n_nt


def _equilibrate_ops(alive, theta, a, r_max, rcond, newton_solver,
                     cg_tol, cg_maxiter, pos_shape):
    """Operators shared by the monolithic and chunked paths.

    Extracted so there is exactly ONE implementation of each step. Two copies
    that agree today are two copies that disagree after the next edit, and the
    bit-identity gate would then be checking a coincidence rather than a
    property.
    """
    amask = alive[:, None]
    energy = lambda p: field_morse_energy(p, alive, theta, a, r_max)
    grad_fn = lambda p: jax.grad(energy)(p) * amask
    resid = lambda p: jnp.abs(grad_fn(p)).max()

    def armijo(p, g, e0, lr):
        gn2 = (g * g).sum()

        def cond(state):
            step, i = state
            ok = energy(p - step * g) <= e0 - 1e-4 * step * gn2
            return (~ok) & (i < 50)

        def body(state):
            step, i = state
            return step * 0.5, i + 1

        return jax.lax.while_loop(cond, body, (lr, 0))[0]

    hess_fn = jax.hessian(lambda q: energy(q.reshape(pos_shape)))
    gflat = lambda qf: grad_fn(qf.reshape(pos_shape)).ravel()
    amask_flat = jnp.repeat(alive, 2)

    def newton_dir_pinv(q):
        return jnp.linalg.pinv(hess_fn(q.ravel()), rcond=rcond) @ gflat(q.ravel())

    def newton_dir_cg(q):
        qf = q.ravel()
        Z = rigid_modes(q, alive)
        Av = lambda v: -jax.jvp(gflat, (qf,), (v,))[1]
        M = _projected_operator(Av, Z, amask_flat)
        rhs = project_out(Z, gflat(qf) * amask_flat)
        d, _ = jax.scipy.sparse.linalg.cg(M, rhs, tol=cg_tol, atol=0.0,
                                          maxiter=cg_maxiter)
        return d

    if newton_solver not in ("cg", "pinv"):
        raise ValueError(
            f"unknown newton_solver {newton_solver!r}; expected 'cg' or 'pinv'")
    newton_dir = newton_dir_cg if newton_solver == "cg" else newton_dir_pinv

    def descent_step(p, lr):
        """One Armijo-backtracking descent iteration -> (pos, lr, residual)."""
        g = grad_fn(p)
        step = armijo(p, g, energy(p), lr)
        p_new = p - step * g
        return p_new, jnp.minimum(step * 2.0, 1.0), resid(p_new)

    def newton_step(q, res):
        """One damped projected Newton iteration -> (pos, residual, improved)."""
        d = newton_dir(q).reshape(pos_shape) * amask

        def bt_cond(s):
            t, i = s
            return (resid(q - t * d) >= res) & (i < 30)

        def bt_body(s):
            t, i = s
            return t * 0.5, i + 1

        t, _ = jax.lax.while_loop(bt_cond, bt_body, (1.0, 0))
        q_new = q - t * d
        res_new = resid(q_new)
        take = res_new < res
        return (jnp.where(take, q_new, q), jnp.where(take, res_new, res), take)

    return resid, descent_step, newton_step


@partial(jax.jit, static_argnames=("a", "r_max", "max_descent", "max_newton",
                                   "newton_solver"))
def equilibrate(pos, alive, theta, a=A_DEFAULT, r_max=R_MAX_DEFAULT,
                tol=1e-12, max_descent=5000, max_newton=100, newton_tol=1e-4,
                lr0=0.01, rcond=1e-8, newton_solver="cg", cg_tol=1e-10,
                cg_maxiter=400):
    """Relax to a *genuine* mechanical equilibrium, to machine precision.

    Returns ``(x_star, residual, n_iter, converged)`` where ``residual`` is
    ``max|F|``. **Always check ``converged``** — implicitly differentiating a
    point that is not a fixed point is not a sensitivity of anything.

    Two stages, because no single method does both jobs well:

    1. **Armijo-backtracking descent** to get into the basin (down to
       ``newton_tol``). Every accepted step strictly decreases the energy, so —
       unlike a clipped fixed step — no oscillation can be a fixed point of the
       iteration.
    2. **Damped projected Newton** to polish. Near the minimum, ``x ← x − H⁺∇E``
       converges quadratically and reaches ~1e-14 in a handful of steps. The
       step is damped because ``newton_tol`` is a guess about where the quadratic
       basin starts, and a guess should degrade rather than fail: handing over at
       1e-2 on a heterogeneous θ field gives a step too large to help, and an
       undamped Newton then *freezes* at the handoff residual instead of
       converging slowly.

    ``newton_solver`` picks how the Newton direction is obtained. Both take the
    step in the complement of the null space — H is exactly singular here (the
    rigid modes, plus a trivial mode per padded cell), which is the same gauge
    choice the sensitivity engine makes, for the same reason.

    - ``"cg"`` (default) — **matrix-free**: projected conjugate gradients on
      Hessian-vector products. Never forms H. O(N) memory, and the only path
      that survives scale.
    - ``"pinv"`` — forms the dense (2N × 2N) Hessian and pseudo-inverts it.
      O(N²) memory, O(N³) time; obviously correct, and kept as the reference the
      CG path is gated against.

    Why the default is matrix-free
    ------------------------------
    The dense path's real cost is **O(N³) memory, not O(N²)** — and that is worse
    than it looks. The Hessian itself is only (2N × 2N) — 51 MB at N=1261. But
    ``jax.hessian`` differentiates *through* the O(N²) pair matrix inside
    :func:`field_morse_energy`, producing a ``(2N, N, N)`` intermediate: **32 GB
    at N=1261, 348 GB at N=2791**. Measured on a 16 GB card: ``pinv`` OOMs at
    **N=1261** while ``cg`` runs, and where both fit (N=331) CG is ~3.6x faster.
    CG needs only Hessian-vector products, so it never builds that intermediate.

    Necessary, not sufficient — and the next wall is not where I expected
    ----------------------------------------------------------------------
    The prediction was that removing the solver wall would expose
    :func:`field_morse_energy`'s O(N²) pair matrix (dying ~N=20 000, fixable with
    the neighbour lists ``jax_morpho.scale`` already implements). **Measurement
    says otherwise: the next wall is the descent stage.** At N=1261 CG-Newton
    still fails to converge (``max|F| ~ 1``, 5000 descent iterations exhausted),
    because gradient descent needs O(N) iterations to relax the long-wavelength
    "breathing" mode of a large blob — the condition number grows with system
    size, and a hexagonal lattice at ``spacing = r_eq`` is *not* the Morse
    equilibrium (second neighbours at √3 ≈ 1.73 sit inside ``r_max = 1.8`` and
    pull), so the whole blob must relax collectively.

    Nor can the descent stage simply be dropped: ``newton_tol = 1e2``
    ("always Newton, damped") **fails** — it stalls after 10–20 iterations at
    ``max|F| ~ 0.9``. Far from the minimum H is not positive-definite, so
    ``H⁻¹∇E`` is not a descent direction and no amount of damping rescues it.
    Newton is mesh-independent but only *inside* the basin; something has to get
    it there.

    So organism scale needs a better globalisation (accelerated/preconditioned
    descent — FIRE or Nesterov, standard in molecular statics, O(√N) instead of
    O(N)) *before* the energy's O(N²) matters. Note the pipeline may dodge this:
    individuals are small perturbations of a reference genome whose equilibrium
    is already known. **Warm-starting from that reference would be a cheat**,
    though — it biases which basin development lands in and would suppress
    exactly the multistability §3c measures. Recorded in docs/DESIGN.md §2E.

    Why the handoff exists at all
    -----------------------------
    Neither stage can do the whole job. Armijo descent alone stalls near
    ``max|F| ~ 1e-7`` and then *drifts upward*: the per-step energy decrease
    (~1e-14) falls below float64's *absolute* resolution of the energy itself
    (E ~ −18, hence ~2e-15), so the sufficient-decrease test degenerates into a
    comparison of roundoff noise and backtracks to a zero step. Newton alone is
    unsafe from a random initial blob. So the line search runs only while its
    decisions are numerically meaningful, and hands over to Newton — which
    tests gradients, never energy differences — for the last five orders of
    magnitude. That 1e-7 stall is not academic: gate #1 divides an equilibrium
    difference by ``2·eps``, so a 1e-7 equilibrium would swamp the finite-
    difference reference with ~5% noise and the gate would measure nothing.

    Why not ``center_based.relax``
    ------------------------------
    That function takes a fixed number of steps at a fixed learning rate with a
    smooth saturating clip on the step. When the effective step exceeds the
    stability limit ``2/λ_max`` of the stiffest mode, the clip does not diverge —
    it *stabilises the instability into a period-2 limit cycle*. The iterate
    then oscillates between two points forever: the energy alternates, ``|∇E|``
    pins at a constant nonzero value, and running more steps changes nothing. It
    looks converged and is not. (Measured on a 12-cell blob: ``|∇E| = 3.34`` at
    200 steps and still 3.34 at 100 000, while ``|p_{k+2} − p_k| = 1e-16`` and
    ``|p_{k+1} − p_k| = 0.05``.) That silent non-convergence — not unroll length
    — is what broke autodiff through the relaxation; see docs/DESIGN.md §1.
    """
    resid, descent_step, newton_step = _equilibrate_ops(
        alive, theta, a, r_max, rcond, newton_solver, cg_tol, cg_maxiter,
        pos.shape)

    # -- stage 1: Armijo-backtracking descent into the basin ---------------
    def gd_cond(state):
        _, _, it, res = state
        return (res > newton_tol) & (it < max_descent)

    def gd_body(state):
        p, lr, it, _ = state
        p_new, lr_new, res_new = descent_step(p, lr)
        return p_new, lr_new, it + 1, res_new

    p, _, it_gd, res = jax.lax.while_loop(
        gd_cond, gd_body, (pos, lr0, 0, resid(pos)))

    # -- stage 2: damped projected Newton polish --------------------------
    def nt_cond(state):
        _, it, res, moving = state
        return (res > tol) & (it < max_newton) & moving

    def nt_body(state):
        q, it, res, _ = state
        q_new, res_new, take = newton_step(q, res)
        return q_new, it + 1, res_new, take

    x, it_nt, res, _ = jax.lax.while_loop(nt_cond, nt_body, (p, 0, res, True))
    return x, res, it_gd + it_nt, res <= tol


# ---------------------------------------------------------------------------
# Chunked / resumable equilibration
# ---------------------------------------------------------------------------
#
# `equilibrate` is a `lax.while_loop` with a DATA-DEPENDENT trip count. Under
# `vmap` that runs the whole batch to the slowest member's iteration count, so
# per-tick cost is unbounded — the blocker `Planktonica/docs/DESIGN_larval.md`
# §2 records for larval development.
#
# The chunked path advances EXACTLY `steps` iterations via `fori_loop`, so the
# trip count is static and vmap-safe. A member that has already converged does
# NOT exit — it takes no-op steps. Exiting would either reintroduce the
# data-dependent count or corrupt its neighbours in the batch, and that is the
# one choice the whole design turns on.
#
# Both paths call the same `descent_step` / `newton_step`, so chunking to
# completion is BIT-IDENTICAL to the monolithic call rather than merely close.
# `tests/test_chunked_equilibrate.py` gates that, and it is a check that can
# fail: dropping `lr` from the carried state, or letting the descent→Newton
# handoff land on a different iteration, each break it.


def equilibrate_init(pos, alive, theta, a=A_DEFAULT, r_max=R_MAX_DEFAULT,
                     newton_tol=1e-4, lr0=0.01) -> EquilibrateState:
    """Fresh state for a chunked relaxation.

    `phase` is chosen from the initial residual, so a body already inside the
    quadratic basin starts in Newton — exactly what the monolithic path does
    when its descent `while_loop` finds the condition false on entry.
    """
    resid, _, _ = _equilibrate_ops(alive, theta, a, r_max, 1e-8, "cg",
                                   1e-10, 400, pos.shape)
    res = resid(pos)
    return EquilibrateState(
        pos=pos,
        lr=jnp.asarray(lr0, dtype=pos.dtype),
        res=res,
        n_gd=jnp.asarray(0, dtype=jnp.int32),
        n_nt=jnp.asarray(0, dtype=jnp.int32),
        phase=jnp.where(res > newton_tol, 0, 1).astype(jnp.int32),
    )


@partial(jax.jit, static_argnames=("steps", "a", "r_max", "max_descent",
                                   "max_newton", "newton_solver"))
def equilibrate_chunk(state, alive, theta, steps=64,
                      a=A_DEFAULT, r_max=R_MAX_DEFAULT, tol=1e-12,
                      max_descent=5000, max_newton=100, newton_tol=1e-4,
                      rcond=1e-8, newton_solver="cg", cg_tol=1e-10,
                      cg_maxiter=400):
    """Advance a relaxation by EXACTLY `steps` iterations. Constant cost.

    Safe under `vmap`: every member executes `steps` iterations regardless of
    its own progress, and a finished member's steps are no-ops.

    Returns an updated `EquilibrateState`. **Done is not converged**: `phase == 2`
    only says the run stopped, which it also does on exhausting `max_descent` or
    stalling in Newton. Converged is `phase == 2 and res <= tol`. Conflating
    them is the failure mode `center_based.relax` taught us to distrust — see
    `equilibrate`'s docstring.
    """
    _, descent_step, newton_step = _equilibrate_ops(
        alive, theta, a, r_max, rcond, newton_solver, cg_tol, cg_maxiter,
        state.pos.shape)

    def one(_, st):
        def do_descent(st):
            p_new, lr_new, res_new = descent_step(st.pos, st.lr)
            n_gd = st.n_gd + 1
            # The monolithic loop re-tests `(res > newton_tol) & (it <
            # max_descent)` at the top of the next iteration; this is the same
            # predicate at the same point, so the handoff lands identically.
            to_newton = res_new <= newton_tol
            exhausted = n_gd >= max_descent
            phase = jnp.where(to_newton, 1,
                              jnp.where(exhausted, 2, 0)).astype(jnp.int32)
            return st._replace(pos=p_new, lr=lr_new, res=res_new,
                               n_gd=n_gd, phase=phase)

        def do_newton(st):
            q_new, res_new, took = newton_step(st.pos, st.res)
            n_nt = st.n_nt + 1
            # `took` is the stall signal: even a 2^-30 step failed to improve.
            done = (res_new <= tol) | (n_nt >= max_newton) | (~took)
            return st._replace(pos=q_new, res=res_new, n_nt=n_nt,
                               phase=jnp.where(done, 2, 1).astype(jnp.int32))

        def do_done(st):
            # NOT an early exit. Under vmap the trip count must stay static, so
            # a finished member idles rather than leaving the loop.
            return st

        return jax.lax.switch(st.phase, [do_descent, do_newton, do_done], st)

    return jax.lax.fori_loop(0, steps, one, state)


def equilibrate_chunked(pos, alive, theta, steps=64, max_chunks=2000, **kw):
    """Run `equilibrate_chunk` to completion. Same answer as `equilibrate`.

    Exists mainly to be compared against `equilibrate` bit for bit; production
    callers want to hold the state and spend one chunk per tick instead.
    """
    init_keys = ("a", "r_max", "newton_tol", "lr0")
    st = equilibrate_init(pos, alive, theta,
                          **{k: kw[k] for k in init_keys if k in kw})
    chunk_kw = {k: v for k, v in kw.items() if k != "lr0"}
    for _ in range(max_chunks):
        if int(st.phase) == 2:
            break
        st = equilibrate_chunk(st, alive, theta, steps=steps, **chunk_kw)
    return st


#: Contact cutoff separating first neighbours (~r_eq = 1.0) from second
#: neighbours (~√3 r_eq = 1.73) in a hexagonal packing. Any value in the gap
#: works; see :func:`contact_topology` for why the gap is what matters.
CONTACT_CUTOFF = 1.3


def contact_topology(pos, alive=None, cutoff=CONTACT_CUTOFF):
    """Which cells touch which — a fingerprint of the developmental *basin*.

    Returns the set of contacting pairs (centres closer than ``cutoff``) as a
    frozenset of sorted index pairs. Two equilibria with the same contact set are
    the same packing, deformed; a changed contact set means a neighbour exchange
    — a different developmental outcome.

    Why this exists. The Morse energy landscape is **multistable**, so the
    genotype→phenotype map is only piecewise smooth: a large enough genetic
    perturbation pushes the tissue over a barrier into a different packing, and
    the phenotype jumps *discontinuously*. Any local object — the developmental
    Jacobian, and so G — describes the response **within a basin** and is silent
    about crossings between them.

    Why contacts and not a Delaunay triangulation
    ---------------------------------------------
    The obvious fingerprint is the Delaunay edge set, and on this system it is
    **wrong**. A hexagonal lattice is the maximally *cocircular* configuration —
    it is dense with quadruples of points on a common circle — so its Delaunay
    triangulation is degenerate and flips diagonals under infinitesimal
    perturbation, with no rearrangement of anything physical. Measured on a
    19-cell blob at σ=0.05 against an unambiguous ground truth (equilibria that
    moved by ~20x the typical distance): Delaunay flagged 23/120 individuals of
    which **21 were false positives**. Contacts at ``cutoff=1.3`` flagged 8, all
    real jumps included. (``center_based.interior_side_counts`` uses Delaunay
    legitimately — it measures *disordered* packings, where cocircularity is
    measure-zero rather than the norm.)

    The cutoff is not tuned to an answer; it just has to fall in the gap between
    first and second neighbours. Pairs sitting near it can flip without a real
    rearrangement, so this over-reports slightly — treat a changed contact set as
    "worth a look", and a large jump in the form itself as proof.

    Indices are **original cell indices**, not compressed ones
    ---------------------------------------------------------
    Masking to live cells renumbers them, and a renumbered contact set is not
    comparable to anything. Two individuals with the same padded layout but
    *different* slots empty — which is exactly what growth produces — would each
    report a pair ``(3, 4)`` meaning different original cells, and comparing the
    sets would silently compare different tissues. So the mask is applied to the
    *pairs*, and the original indices are kept. (Found by Copilot on PR #3.)

    Host-side, so not jittable or vmappable — a diagnostic, not part of the
    differentiable path.
    """
    P = np.asarray(pos)
    live = (np.ones(len(P), bool) if alive is None
            else np.asarray(alive) > 0.5)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    i, j = np.triu_indices(len(P), 1)
    keep = (d[i, j] < cutoff) & live[i] & live[j]
    return frozenset((int(a), int(b)) for a, b in zip(i[keep], j[keep]))


def develop(theta, pos0, alive, a=A_DEFAULT, r_max=R_MAX_DEFAULT, tol=1e-12):
    """The developmental map θ → x*: relax ``pos0`` to mechanical equilibrium.

    Raises if the equilibrium was not reached, rather than silently handing a
    non-fixed-point to the sensitivity engine.
    """
    x, res, _, ok = equilibrate(pos0, alive, theta, a, r_max, tol)
    if not bool(ok):
        raise RuntimeError(
            f"development did not reach equilibrium: max|F| = {float(res):.3e} "
            f"> tol = {tol:.3e}; sensitivity at this point is meaningless")
    return x
