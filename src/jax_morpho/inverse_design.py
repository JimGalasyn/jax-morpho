"""Gradient-based inverse design of tissue morphology.

The center-based relaxation (center_based.relax) is differentiable, so we
can optimize the *generative* parameters of a tissue to hit a target
morphology by autodiff through the entire mechanical relaxation — the
inverse problem of morphogenesis.

Here the generative parameters are an anisotropic confinement field (a
morphogen / boundary condition): V_ext = 1/2 (kx x^2 + ky y^2).  Cells pack
under Morse adhesion inside the field; its anisotropy sets the tissue's
aspect ratio and its strength sets the size.  ``fit_morphology`` recovers
the field that produces a target (size, aspect) via gradient descent, and
demonstrably beats gradient-free search at equal budget (see
tests/test_inverse_design.py and examples/demo_inverse_design.py).

A hand-rolled Adam keeps this free of an optax dependency.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from jax_morpho.center_based import (
    D_DEFAULT, A_DEFAULT, R_EQ_DEFAULT, R_MAX_DEFAULT, gyration_morphology)


@partial(jax.jit, static_argnums=())
def confinement_energy(pos, params, D=D_DEFAULT, a=A_DEFAULT,
                       r_eq=R_EQ_DEFAULT, r_max=R_MAX_DEFAULT):
    """Morse pair energy + anisotropic confinement V=1/2(kx x^2 + ky y^2).

    ``params`` = [log kx, log ky] (log-parameterized so k stays positive).
    """
    kx, ky = jnp.exp(params[0]), jnp.exp(params[1])
    diff = pos[:, None, :] - pos[None, :, :]
    r = jnp.sqrt((diff * diff).sum(-1) + 1e-9)
    raw = (1.0 - jnp.exp(-a * (r - r_eq))) ** 2 - 1.0
    shift = (1.0 - jnp.exp(-a * (r_max - r_eq))) ** 2 - 1.0
    n = pos.shape[0]
    pair = (r < r_max) * (1.0 - jnp.eye(n))
    e_pair = 0.5 * D * ((raw - shift) * pair).sum()
    e_ext = 0.5 * (kx * pos[:, 0] ** 2 + ky * pos[:, 1] ** 2).sum()
    return e_pair + e_ext


@partial(jax.jit, static_argnums=(2,))
def relax_in_field(pos, params, n_steps=200, lr=0.02, max_disp=0.1):
    """Relax cells to equilibrium under the parameterized field."""
    g = jax.grad(confinement_energy)

    def step(P, _):
        s = lr * g(P, params)
        norm = jnp.sqrt((s * s).sum(-1, keepdims=True) + 1e-12)
        s = s * (max_disp / (norm + max_disp))
        return P - s, None

    P, _ = jax.lax.scan(step, pos, None, length=n_steps)
    return P


def adam(loss_fn, p0, lr=0.1, n_steps=40, b1=0.9, b2=0.999, eps=1e-8):
    """Minimal Adam on a scalar autodiff loss.  Returns (params, history)."""
    vg = jax.jit(jax.value_and_grad(loss_fn))
    p = jnp.asarray(p0)
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)
    history = []
    for t in range(1, n_steps + 1):
        L, g = vg(p)
        history.append(float(L))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        p = p - lr * mhat / (jnp.sqrt(vhat) + eps)
    return p, history


def morphology_loss(params, pos0, size_target, aspect_target, relax_steps=200):
    """Normalized (size, aspect) mismatch after relaxation under ``params``."""
    size, aspect = gyration_morphology(relax_in_field(pos0, params, relax_steps))
    return (((size - size_target) / size_target) ** 2
            + ((aspect - aspect_target) / aspect_target) ** 2)


def fit_morphology(pos0, size_target, aspect_target, p0=None,
                   lr=0.1, n_steps=40, relax_steps=200):
    """Optimize the confinement field to hit (size_target, aspect_target).

    Returns (params, history) where history is the loss per iteration.
    """
    if p0 is None:
        p0 = jnp.array([np.log(0.12), np.log(0.12)], dtype=jnp.float32)

    def loss(p):
        return morphology_loss(p, pos0, size_target, aspect_target, relax_steps)

    return adam(loss, p0, lr=lr, n_steps=n_steps)
