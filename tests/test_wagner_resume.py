"""`DevResult.s_T` — resuming a partial developmental settle.

Required by `Planktonica/docs/DESIGN_larval.md` §2, and named in
`DESIGN_seam.md` §5 as one of two changes to promote upstream immediately.
`develop` already accepts `s0`, so resuming needs no new machinery — only the
final state, which was previously discarded into `_`.

The distinction these tests exist to pin: **`s_T` composes over any chunking;
`phenotype`, `xi` and `stable` do not.** Those three are statistics over the
trailing `window` of *this call's* trajectory, so a chunk shorter than `window`
reports a window it never had. Getting that wrong would make a larva's stability
test read a window that does not exist, and it would look like noise rather than
a bug.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_morpho.evodevo import wagner_grn as wg


@pytest.fixture(scope="module")
def net():
    return wg.random_network(jax.random.PRNGKey(0), n=12)


# --------------------------------------------------------------------------- #
# s_T is the resume point
# --------------------------------------------------------------------------- #

def test_s_T_is_the_final_state_not_the_windowed_mean(net):
    """Guards against `s_T` being wired to `phenotype` by mistake."""
    r = wg.develop(net.W, net.s0, devsteps=40)
    assert r.s_T.shape == net.s0.shape
    # One more step from s_T must equal the 41st state of the unsplit run.
    one_more = wg.develop(net.W, r.s_T, devsteps=1, window=1)
    direct = wg.develop(net.W, net.s0, devsteps=41, window=1)
    assert np.allclose(np.asarray(one_more.s_T), np.asarray(direct.s_T),
                       atol=0, rtol=0)


@pytest.mark.parametrize("chunks", [[20, 20], [10, 10, 10, 10], [1] * 40,
                                    [37, 3], [39, 1]])
def test_s_T_composes_over_any_chunking(net, chunks):
    """Splitting devsteps must reproduce the unsplit trajectory exactly.

    Bit-exact, not close: the recurrence is deterministic and the chunk boundary
    changes nothing about the arithmetic, unlike the jax-morpho `equilibrate`
    case where two different loop constructs are fused differently by XLA.
    """
    assert sum(chunks) == 40
    ref = wg.develop(net.W, net.s0, devsteps=40)

    s = net.s0
    for n in chunks:
        s = wg.develop(net.W, s, devsteps=n, window=1).s_T

    assert np.array_equal(np.asarray(s), np.asarray(ref.s_T)), (
        f"chunking {chunks} drifted: max|Δ| = "
        f"{np.abs(np.asarray(s) - np.asarray(ref.s_T)).max():.3e}"
    )


# --------------------------------------------------------------------------- #
# ...and the statistics do NOT
# --------------------------------------------------------------------------- #

def test_windowed_stats_agree_when_the_last_chunk_covers_the_window(net):
    """The condition `DESIGN_larval.md` relies on: LARVAL_STEPS >= DEFAULT_WINDOW."""
    window = 10
    ref = wg.develop(net.W, net.s0, devsteps=40, window=window)

    mid = wg.develop(net.W, net.s0, devsteps=30, window=window)
    tail = wg.develop(net.W, mid.s_T, devsteps=10, window=window)
    # LARVAL_STEPS (10) >= DEFAULT_WINDOW (10): the final chunk covers the window.

    assert np.allclose(np.asarray(tail.phenotype), np.asarray(ref.phenotype),
                       atol=1e-6)
    assert np.isclose(float(tail.xi), float(ref.xi), atol=1e-6)
    assert bool(tail.stable) == bool(ref.stable)


def test_windowed_stats_are_wrong_when_the_last_chunk_is_shorter(net):
    """A chunk shorter than `window` reports a window it never had.

    **Only observable while the trajectory is still moving.** Once the network
    settles to a fixed point every state in any window is identical, so the
    windowed mean is the same at any chunk length and a converged run hides the
    bug entirely. The first version of this test split at step 37, where
    `xi = 8.9e-16` — long settled — and it failed by *matching*.

    That is exactly the larval case though: a partially-developed larva has not
    settled, which is the whole reason it is still a larva. So the constraint
    bites precisely where it is going to be used, and is invisible in the easy
    case.

    "Still moving" is measured RELATIVE TO `DEFAULT_EPS`, the model's own
    settling threshold, and the split point is searched for rather than
    hardcoded. Both used to be absolute: split at a literal 6 steps, guarded by
    `xi > 1e-2`, chosen because this fixture had `xi = 1.9e-01` there. That is a
    property of one network, and the network depends on the dtype — under
    `jax_enable_x64` the same PRNGKey yields a different `W` whose `xi` never
    exceeds 1.8e-03 at any split. The guard then fired in x64 and passed in x32,
    reporting a dtype difference as a windowing regression.

    But `xi = 1.8e-03` is still 18x the `1e-04` at which this model calls a
    network settled, and the short-chunk error there is ~1e-03 — a thousand
    times the comparison tolerance below. The trajectory was moving fine; only
    the hardcoded constant disagreed. Scaling the guard to `DEFAULT_EPS` says
    what was actually meant, in the units the model defines.
    """
    window = 4
    moving = 10.0 * wg.DEFAULT_EPS          # an order of magnitude from settled
    devsteps = next((d for d in range(window + 2, 40)
                     if float(wg.develop(net.W, net.s0, devsteps=d,
                                         window=window).xi) > moving), None)
    assert devsteps is not None, (
        "no split point found where the trajectory is still moving; this "
        "fixture settles too fast to exercise the short-chunk case at all")

    ref = wg.develop(net.W, net.s0, devsteps=devsteps, window=window)
    assert float(ref.xi) > moving, "trajectory must still be moving, or this is vacuous"

    head = devsteps - 2
    mid = wg.develop(net.W, net.s0, devsteps=head, window=window)
    tail = wg.develop(net.W, mid.s_T, devsteps=2, window=window)

    # `scan` yields only 2 states, so `traj[-4:]` is 2 rows, not 4.
    assert not np.allclose(np.asarray(tail.phenotype),
                           np.asarray(ref.phenotype), atol=1e-6), (
        "short final chunk unexpectedly matched — if the windowing changed, "
        "this constraint may no longer hold and the docstring is stale"
    )


# --------------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------------- #

def test_existing_fields_are_unchanged(net):
    """Appending a field must not disturb what was already there."""
    r = wg.develop(net.W, net.s0)
    assert r._fields == ("phenotype", "stable", "xi", "s_T"), (
        "s_T must stay LAST — every consumer reads DevResult by attribute, so a "
        "trailing field is additive, but inserting one would break unpacking"
    )
    assert r.phenotype.shape == net.s0.shape
    assert r.xi.shape == ()
    assert r.stable.shape == ()


def test_develop_pop_carries_s_T_batched(net):
    """The vmapped path must expose s_T too, or larvae cannot resume in batch."""
    pop = 5
    W = jnp.broadcast_to(net.W, (pop,) + net.W.shape)
    s0 = jnp.broadcast_to(net.s0, (pop,) + net.s0.shape)
    # develop_pop's in_axes is a 6-tuple, so every positional arg must be given.
    res = wg.develop_pop(W, s0, wg.DEFAULT_A, wg.DEFAULT_DEVSTEPS,
                         wg.DEFAULT_WINDOW, wg.DEFAULT_EPS)
    assert res.s_T.shape == (pop, net.s0.shape[0])
    # Identical inputs -> identical rows; a broken vmap axis shows up here.
    assert np.allclose(np.asarray(res.s_T[0]), np.asarray(res.s_T[-1]))
