"""Center-based differentiable tissue engine (JAX).

Cells are points; a Morse pairwise potential gives soft repulsion (cells
resist overlap) plus short-range adhesion (cells stick).  Relaxation is
gradient descent on the autodiff energy; growth divides a cell into two
nearby points.  Fixed-size padded arrays with an ``alive`` mask keep the
whole thing jit-friendly and map cleanly onto GPU data layout.

Why center-based (rather than a vertex model): the representation vectorizes
cleanly, stays differentiable through cell division, and scales — the prior
art (JAX-MD, Deshpande et al. 2024) shows this is the path to GPU/organism-
scale differentiable morphogenesis, where vertex-model topology operations
(T1, mitosis) are GPU-hostile.

Emergent calibration result (see tests/test_center_based.py): coupling
proliferation to relaxation reproduces the real-epithelium (Gibson 2006)
polygon-side distribution, tunable between a random (Poisson-Voronoi) and a
crystalline (hexagonal) packing by the relaxation-per-division ratio.  This
is the distribution neither passive vertex relaxation nor pure centroidal
ordering could hit.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import Delaunay

# Default Morse parameters: r_eq is the preferred cell spacing, r_max the
# interaction cutoff, D the adhesion well depth, a the repulsion steepness.
D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT = 1.0, 2.5, 1.0, 1.8


# ---------------------------------------------------------------------------
# Pairwise energy + autodiff relaxation
# ---------------------------------------------------------------------------

@jax.jit
def morse_energy(pos, alive, D, a, r_eq, r_max):
    """Total Morse pair energy over alive cells within the cutoff.

    pos: (N,2) positions; alive: (N,) in {0,1}.  Dead cells and pairs
    beyond r_max contribute nothing; the diagonal is excluded.
    """
    diff = pos[:, None, :] - pos[None, :, :]
    r = jnp.sqrt((diff * diff).sum(-1) + 1e-9)
    raw = D * ((1.0 - jnp.exp(-a * (r - r_eq))) ** 2 - 1.0)
    shift = D * ((1.0 - jnp.exp(-a * (r_max - r_eq))) ** 2 - 1.0)
    u = raw - shift
    n = pos.shape[0]
    pair = alive[:, None] * alive[None, :] * (r < r_max) * (1.0 - jnp.eye(n))
    return 0.5 * (u * pair).sum()


@partial(jax.jit, static_argnums=(6, 7, 8))
def relax(pos, alive, D, a, r_eq, r_max, n_steps=200, lr=0.02, max_disp=0.1):
    """Fixed-step gradient descent on the Morse energy, toward equilibrium.

    Uses a smooth, always-finite step clip (no ``jnp.where``) so the whole
    relaxation stays autodiff-differentiable w.r.t. the force parameters —
    the inverse-design hook.

    .. warning::
       **This does not converge to a fixed point, and more steps do not help.**
       When the effective step exceeds the stability limit ``2/λ_max`` of the
       stiffest mode, the clip does not diverge — it stabilises the instability
       into a *period-2 limit cycle*. The iterate then orbits between two points
       forever, so it looks converged (bounded energy, constant gradient norm)
       while ``|∇E|`` never falls. Measured on a 12-cell blob at the defaults:
       ``|∇E| = 3.34`` at ``n_steps=200`` and still 3.34 at 100 000, with
       ``|p_{k+2} − p_k| = 1e-16`` against ``|p_{k+1} − p_k| = 0.05``.

       This is fine for what it is used for here — growth-and-relax packing
       (:func:`grow_relax`), where the relaxation-per-division ratio is a tuning
       knob and the epithelial topology statistics are calibrated against this
       exact behaviour. It is **not** usable where a genuine equilibrium is
       required, notably as input to implicit differentiation. For that, use
       :func:`jax_morpho.evodevo.mechanical.equilibrate`, which reports its
       residual and reaches ``max|F| ~ 1e-14``. See docs/DESIGN.md §1.
    """
    grad_fn = jax.grad(morse_energy)
    amask = alive[:, None]

    def step(p, _):
        g = grad_fn(p, alive, D, a, r_eq, r_max) * amask
        s = lr * g
        norm = jnp.sqrt((s * s).sum(-1, keepdims=True) + 1e-12)
        s = s * (max_disp / (norm + max_disp))     # smooth saturating clip
        return p - s, None

    p, _ = jax.lax.scan(step, pos, None, length=n_steps)
    return p


# ---------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------

def divide_cells(pos, alive, rng, frac, r_eq=R_EQ_DEFAULT, offset=0.3):
    """Divide a fraction of alive cells into free slots.

    A dividing cell's point splits into two, offset symmetrically along a
    random axis.  Operates on numpy arrays (host side); returns new
    (pos, alive) with the same fixed size.
    """
    pos = np.array(pos)
    alive = np.array(alive)
    live_idx = np.where(alive > 0.5)[0]
    dead_idx = np.where(alive < 0.5)[0]
    n_div = min(int(frac * len(live_idx)), len(dead_idx))
    if n_div == 0:
        return pos, alive
    parents = rng.choice(live_idx, n_div, replace=False)
    for k, p in enumerate(parents):
        d = dead_idx[k]
        theta = rng.uniform(0, 2 * np.pi)
        off = offset * r_eq * np.array([np.cos(theta), np.sin(theta)])
        pos[d] = pos[p] + off
        pos[p] = pos[p] - off
        alive[d] = 1.0
    return pos, alive


def grow_relax(n_max, n_start, target, relax_steps,
               D=D_DEFAULT, a=A_DEFAULT, r_eq=R_EQ_DEFAULT, r_max=R_MAX_DEFAULT,
               divide_frac=0.5, seed=0):
    """Grow from a small blob to ``target`` cells, alternating division and
    ``relax_steps`` of relaxation.  Small ``relax_steps`` (proliferation-
    dominated) yields a disordered packing; large ``relax_steps``
    (relaxation-dominated) yields a crystalline one.  Returns (pos, alive)
    numpy arrays of shape (n_max, 2) and (n_max,).
    """
    if target > n_max:
        raise ValueError(
            f"target ({target}) exceeds n_max ({n_max}); no free slots to "
            f"divide into would leave the growth loop unable to progress")
    rng = np.random.default_rng(seed)
    pos = np.zeros((n_max, 2), np.float32)
    alive = np.zeros(n_max, np.float32)
    pos[:n_start] = rng.normal(0, r_eq, size=(n_start, 2)).astype(np.float32)
    alive[:n_start] = 1.0
    pos = np.array(relax(jnp.asarray(pos), jnp.asarray(alive),
                         D, a, r_eq, r_max, 300))
    while int(alive.sum()) < target:
        pos, alive = divide_cells(pos, alive, rng, divide_frac, r_eq)
        pos = np.array(relax(jnp.asarray(pos), jnp.asarray(alive),
                             D, a, r_eq, r_max, relax_steps))
    return pos, alive


# ---------------------------------------------------------------------------
# Topology measurement (Voronoi side counts via Delaunay degree)
# ---------------------------------------------------------------------------

def interior_side_counts(pos, alive) -> np.ndarray:
    """Polygon-side count of each interior cell (= Delaunay degree), for
    cells not on the convex hull.  Feeds the epithelial topology laws."""
    P = pos[alive > 0.5]
    tri = Delaunay(P)
    nbr = [set() for _ in range(len(P))]
    for s in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    nbr[s[i]].add(int(s[j]))
    hull = set(int(v) for v in tri.convex_hull.ravel())
    interior = [i for i in range(len(P)) if i not in hull]
    return np.array([len(nbr[i]) for i in interior], dtype=int)


@jax.jit
def gyration_morphology(pos):
    """Differentiable tissue shape descriptor from the gyration tensor.

    Returns (size, aspect): ``size`` is the radius of gyration, ``aspect``
    the ratio of principal axes (1.0 = round, >1 = elongated).  Closed-form
    2x2 eigenvalues keep it autodiff-safe.  Used as an inverse-design
    objective (see jax_morpho.inverse_design).
    """
    c = pos - pos.mean(0)
    G = (c.T @ c) / pos.shape[0]
    tr = G[0, 0] + G[1, 1]
    det = G[0, 0] * G[1, 1] - G[0, 1] ** 2
    disc = jnp.sqrt(jnp.maximum((tr / 2) ** 2 - det, 1e-9))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    return jnp.sqrt(l1 + l2), jnp.sqrt(l1 / jnp.maximum(l2, 1e-9))
