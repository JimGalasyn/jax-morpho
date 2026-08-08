"""Tests for the Wagner recurrent GRN (Siegal-Bergman canalization port).

Covers: model correctness (activation range, fixed-point stability),
differentiability of development in the genome W, and the core canalization
signature — mutational robustness rises under stabilizing selection.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jax_morpho.evodevo import wagner_grn as wg


def test_deterministic_xla_flags_appends_idempotently_and_keeps_caller_flags():
    """The GPU-reproducibility mitigation the module docstring points at.

    Worth testing rather than eyeballing because the two ways to get it wrong are
    both silent: clobbering a caller's flags (their setting vanishes, nothing
    errors) and appending twice on a second call (XLA takes the flag either way,
    so the duplicate never surfaces as a failure).
    """
    flag = "--xla_gpu_autotune_level=0"

    assert wg.deterministic_xla_flags() == flag
    assert wg.deterministic_xla_flags(None) == flag
    assert wg.deterministic_xla_flags("") == flag

    kept = wg.deterministic_xla_flags("--xla_force_host_platform_device_count=4")
    assert "--xla_force_host_platform_device_count=4" in kept   # caller's survives
    assert flag in kept

    assert wg.deterministic_xla_flags(kept) == kept             # idempotent
    assert kept.count(flag) == 1


def test_develop_pop_is_callable_with_just_the_batched_arguments():
    """`develop_pop(W, s0)` must work — the defaults are the whole point.

    Bare `jax.vmap(..., in_axes=(0,)*2 + (None,)*4)` requires all six positional
    arguments and raises `ValueError: len(in_axes)=6, len(args)=2` otherwise, so
    a caller reading `develop`'s signature and supplying only the batched inputs
    got an error instead of the defaults. That is exactly what drove one consumer
    to build a duplicate batcher.

    Also pins agreement with the unbatched `develop`, so the wrapper cannot start
    passing the defaults through in the wrong order without failing.
    """
    net = wg.random_network(jax.random.PRNGKey(3), n=8)
    W = jnp.stack([net.W, net.W * 0.5])
    s0 = jnp.stack([net.s0, net.s0])

    batched = wg.develop_pop(W, s0)                       # two args only
    assert batched.phenotype.shape == (2, 8)

    for i, w in enumerate((net.W, net.W * 0.5)):
        one = wg.develop(w, net.s0)
        assert jnp.allclose(batched.phenotype[i], one.phenotype, atol=1e-6)
        assert bool(batched.stable[i]) == bool(one.stable)

    # Explicit non-default kwargs still reach `develop` (order not scrambled).
    short = wg.develop_pop(W, s0, devsteps=3, window=2)
    assert not jnp.allclose(short.phenotype, batched.phenotype, atol=1e-6)


def test_sigma_range_and_fixed_point():
    """sigma is bounded in [-1, 1], strictly interior for finite inputs, and a
    converged network reports itself stable."""
    x = jnp.linspace(-50, 50, 101)
    y = wg.sigma(x)
    assert jnp.all(jnp.abs(y) <= 1.0)               # bounded everywhere
    moderate = wg.sigma(jnp.linspace(-2, 2, 41))    # interior away from saturation
    assert jnp.all(moderate > -1.0) and jnp.all(moderate < 1.0)
    assert float(wg.sigma(jnp.array(0.0))) == 0.0   # odd, fixed at origin

    founder = wg.find_viable_founder(jax.random.PRNGKey(0))
    res = wg.develop(founder.W, founder.s0)
    assert bool(res.stable)
    assert res.phenotype.shape == (wg.DEFAULT_N,)
    assert jnp.all(jnp.abs(res.phenotype) <= 1.0)
    assert float(res.xi) < wg.DEFAULT_EPS


def test_development_is_differentiable_in_the_genome():
    """Gradients flow genome W -> phenotype: this is what makes the GRN a
    drop-in for the autodiff center-based engine (adoption #1's whole point)."""
    founder = wg.find_viable_founder(jax.random.PRNGKey(1))
    target = jnp.zeros(wg.DEFAULT_N)

    def loss(W):
        return jnp.sum((wg.develop(W, founder.s0).phenotype - target) ** 2)

    g = jax.grad(loss)(founder.W)
    assert g.shape == founder.W.shape
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.sum(jnp.abs(g))) > 0.0   # non-trivial gradient


def test_topology_conserved_under_mutation():
    """Mutation changes interaction strengths but never the wiring (mask)."""
    founder = wg.find_viable_founder(jax.random.PRNGKey(2))
    Wm = wg.mutate(jax.random.PRNGKey(3), founder.W, founder.mask,
                   rate=0.5, sd=1.0)
    zero = founder.mask == 0
    assert jnp.all(Wm[zero] == 0.0)          # no interaction created
    We = wg.single_edge_mutation(jax.random.PRNGKey(4), founder.W,
                                 founder.mask, sd=1.0)
    assert jnp.all(We[zero] == 0.0)
    assert int(jnp.sum(We != founder.W)) == 1  # exactly one edge changed


@pytest.mark.slow
def test_canalization_rise_under_stabilizing_selection():
    """Core Siegal-Bergman result: mutational robustness increases under
    stabilizing selection. Seed 0 gives a large margin (~+0.13); the assertion
    uses a conservative +0.02 floor (all five demo seeds clear +0.03)."""
    key = jax.random.PRNGKey(0)
    k_found, k_evo = jax.random.split(key)
    founder = wg.find_viable_founder(k_found)
    _, hist = wg.evolve(k_evo, founder, pop_size=150, generations=120,
                        stabilizing=True, measure_every=120)
    rise = hist.robustness[-1] - hist.robustness[0]
    assert rise > 0.02, f"expected canalization rise, got {rise:+.3f}"
