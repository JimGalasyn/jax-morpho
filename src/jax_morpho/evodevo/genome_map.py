"""Layer A: the genome → developmental-parameter map (docs/DESIGN.md §2A).

**The genome is the network's input; the network is the developmental map.**
That split is the whole point, and it follows Milocco & Uller's `f`: the map from
genes to developmental parameters is a *fixed, conserved* nonlinear function, and
what evolves is the vector fed into it. Evolving the weights — the evolvability
of development itself — is a separate, later axis (DESIGN.md §2A); here they are
frozen.

The map is a **full MLP, deliberately not affine**. An affine genome→θ map would
make the composed genotype→phenotype map's nonlinearity come entirely from the
mechanics, and the whole question a development-derived G answers — how a
nonlinear developmental map shapes the response to selection — would be
half-assumed away.

How a global genome produces a per-cell field
---------------------------------------------
Every cell reads the same genome but sits somewhere. So each cell's parameters
are ``θ_i = MLP([a, u_i])`` where ``u_i`` is that cell's positional coordinate:
the same conserved network, read against positional information. This is the
standard positional-information picture, and it is what makes θ a *field*
(adhesion and preferred spacing per cell) rather than a couple of globals.

Outputs are squashed into ``[lo, hi]`` by a sigmoid rather than left free. That
is not cosmetic: a θ field with wild adhesion or spacing relaxes into
*disconnected fragments*, and then ∂x*/∂θ genuinely does not exist (see
:func:`~jax_morpho.evodevo.mechanical.hex_blob`). Bounding the field keeps
development inside the regime where the implicit function theorem applies, for
every genome the evolutionary loop might propose.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_morpho.evodevo.mechanical import pack_theta

# Bounds on the per-cell field. Wide enough to give development real room,
# tight enough that no genome can tear the tissue apart.
D_LO, D_HI = 0.5, 1.5           # adhesion well depth
R_EQ_LO, R_EQ_HI = 0.8, 1.2     # preferred spacing


class GRN(NamedTuple):
    """The fixed, conserved developmental map: MLP weights and biases.

    Not evolved in Phase 2 — see the module docstring.
    """
    W1: jnp.ndarray
    b1: jnp.ndarray
    W2: jnp.ndarray
    b2: jnp.ndarray
    W3: jnp.ndarray
    b3: jnp.ndarray


def init_grn(key, n_genes, hidden=16, scale=1.0):
    """Draw a random conserved developmental map.

    ``scale`` sets how nonlinear the map is: it multiplies the first-layer
    weights, so larger values drive the tanh units further from their linear
    region. The default is chosen to be visibly nonlinear while keeping the
    genome→θ response smooth.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    n_in = n_genes + 2                      # genome + (x, y) positional input
    return GRN(
        W1=jax.random.normal(k1, (n_in, hidden)) * (scale / jnp.sqrt(n_in)),
        b1=jnp.zeros(hidden),
        W2=jax.random.normal(k2, (hidden, hidden)) * (1.0 / jnp.sqrt(hidden)),
        b2=jnp.zeros(hidden),
        W3=jax.random.normal(k3, (hidden, 2)) * (1.0 / jnp.sqrt(hidden)),
        b3=jnp.zeros(2),
    )


def _bounded(u, lo, hi):
    """Squash a raw output into (lo, hi) — smooth and strictly interior."""
    return lo + (hi - lo) * jax.nn.sigmoid(u)


def grn_field(genome, coords, grn: GRN):
    """genome ``a`` (n_genes,) + per-cell coords (N,2) → θ field (2N,).

    The same network is applied to every cell, with that cell's position as
    extra input; ``vmap`` over cells makes this one batched matmul.
    """
    def cell(u):
        h = jnp.tanh(jnp.concatenate([genome, u]) @ grn.W1 + grn.b1)
        h = jnp.tanh(h @ grn.W2 + grn.b2)
        return h @ grn.W3 + grn.b3          # (2,) raw

    raw = jax.vmap(cell)(coords)            # (N, 2)
    D = _bounded(raw[:, 0], D_LO, D_HI)
    r_eq = _bounded(raw[:, 1], R_EQ_LO, R_EQ_HI)
    return pack_theta(D, r_eq)


def grn_jacobian(genome, coords, grn: GRN):
    """∂θ/∂a — the (2N, n_genes) Jacobian of the genome→θ map.

    Plain forward-mode autodiff: the map is an explicit function with no solve
    inside it, so there is no fixed point here and nothing implicit to do. The
    implicit machinery is needed only for the developmental *equilibrium*
    downstream (see :mod:`jax_morpho.evodevo.fixed_point`).
    """
    return jax.jacfwd(lambda a: grn_field(a, coords, grn))(genome)
