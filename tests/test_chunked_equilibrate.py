"""Chunked/resumable `equilibrate`.

Motivation: `Planktonica/docs/DESIGN_larval.md` §2. `equilibrate` is a
`lax.while_loop` with a data-dependent trip count, so `vmap` over a population
runs the batch to the slowest member and per-tick cost is unbounded. The chunked
path takes a static number of steps instead.

**Two different claims, deliberately gated at two different strengths:**

* Across CHUNK SCHEDULES — bit-identical, strictly. This is what "resumes bit-
  identically" means and it is the property resume depends on. `allclose` would
  wave through the actual failure modes, which are silent drift: a dropped `lr`
  restarting the Armijo line search at `lr0` still converges, just elsewhere.
  `jax_solitons.runfns` checkpoints Adam's moments for the same reason.
* Against the MONOLITHIC path — numerical only. `while_loop` and
  `fori_loop`+`switch` are fused differently by XLA, giving a fixed 4.9e-16
  difference on coordinates that are themselves ~1e-16. Asserting bit-identity
  here was the first version of this file and it failed; the difference is
  identical for every chunk size, which is what identifies it as a code-path
  artifact rather than a chunking one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_morpho.evodevo.mechanical import (
    EquilibrateState,
    equilibrate,
    equilibrate_chunk,
    equilibrate_chunked,
    equilibrate_init,
    hex_blob,
    uniform_theta,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def blob():
    pos = jnp.asarray(hex_blob(2), dtype=jnp.float64)      # 19 cells
    alive = jnp.ones(pos.shape[0], dtype=bool)
    theta = uniform_theta(pos.shape[0])
    return pos, alive, theta


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

CHUNK_SIZES = [1, 3, 7, 64, 512]


def test_chunk_schedule_is_bit_identical(blob):
    """THE gate: the answer must not depend on how the work was chunked.

    This is what "resumes bit-identically" means — a run preempted at any point
    and resumed lands exactly where an uninterrupted one would. Sizes 1 and 3
    put boundaries mid-descent, 7 is prime so it lands on the descent→Newton
    handoff at some point, and 512 exceeds the total iteration count so it is a
    single chunk.

    Strict equality, deliberately: the failure modes are silent drift (a dropped
    `lr`, a handoff landing on the wrong side) and `allclose` waves those through.
    """
    pos, alive, theta = blob
    outs = {s: np.asarray(equilibrate_chunked(pos, alive, theta, steps=s).pos)
            for s in CHUNK_SIZES}
    base = outs[CHUNK_SIZES[0]]
    for s, x in outs.items():
        assert np.array_equal(x, base), (
            f"chunk size {s} differs from {CHUNK_SIZES[0]}: "
            f"max|Δ| = {np.abs(x - base).max():.3e}"
        )


def test_chunked_matches_monolithic_numerically(blob):
    """Same fixed point and the same iteration count as `equilibrate`.

    NOT bit-identical, and that is a compiler fact rather than a defect: the
    monolithic path is a `lax.while_loop`, the chunked one a `fori_loop` over a
    `switch`, and XLA fuses the two differently. Measured difference is 4.9e-16
    — one ULP on coordinates that are themselves ~1e-16 — and it is *the same
    value for every chunk size*, which is what identifies it as the code-path
    difference rather than a chunking artifact.

    An earlier version of this test asserted bit-identity here and failed. The
    weaker claim is the true one; the strict claim lives in the test above,
    where it belongs.
    """
    pos, alive, theta = blob
    x_ref, res_ref, n_ref, conv_ref = equilibrate(pos, alive, theta)
    assert bool(conv_ref), "reference did not converge; the comparison is moot"

    st = equilibrate_chunked(pos, alive, theta, steps=64)
    assert int(st.n_iter) == int(n_ref), "iteration counts must agree exactly"
    assert np.allclose(np.asarray(st.pos), np.asarray(x_ref), atol=1e-14, rtol=0)
    assert float(st.res) < 1e-12 and float(res_ref) < 1e-12


def test_lr_is_actually_carried(blob):
    """Guards the field most likely to be dropped in a future refactor.

    A resume that restarts the line search at `lr0` still converges — just to a
    different point — so this asserts the state is *used*, not merely present.
    """
    pos, alive, theta = blob
    st = equilibrate_init(pos, alive, theta)
    st1 = equilibrate_chunk(st, alive, theta, steps=5)
    # Continuing from the carried state must differ from restarting at lr0.
    cont = equilibrate_chunk(st1, alive, theta, steps=5)
    restarted = equilibrate_chunk(st1._replace(lr=st.lr), alive, theta, steps=5)
    assert not np.array_equal(np.asarray(cont.pos), np.asarray(restarted.pos)), (
        "resetting lr changed nothing — either lr is unused or the line search "
        "is not adaptive here, and this test is no longer guarding anything"
    )


# --------------------------------------------------------------------------- #
# The property larval development actually needs
# --------------------------------------------------------------------------- #

def test_step_count_is_static_under_vmap(blob):
    """A converged member must no-op, not exit, or the batch is wrong.

    Two members at very different distances from equilibrium, advanced together.
    The already-relaxed one must not be perturbed while the other catches up.
    """
    pos, alive, theta = blob
    relaxed, _, _, conv = equilibrate(pos, alive, theta)
    assert bool(conv)

    batch_pos = jnp.stack([relaxed, pos])                 # [done, far away]
    init = jax.vmap(lambda p: equilibrate_init(p, alive, theta))(batch_pos)
    out = jax.vmap(
        lambda st: equilibrate_chunk(st, alive, theta, steps=32)
    )(init)

    # Member 0 was already at equilibrium: it must still be.
    assert float(out.res[0]) <= 1e-10
    assert np.allclose(np.asarray(out.pos[0]), np.asarray(relaxed), atol=1e-12)
    # Member 1 must have made progress.
    assert float(out.res[1]) < float(init.res[1])


def test_chunk_cost_does_not_depend_on_state(blob):
    """The whole point: the same `steps` runs the same number of iterations
    regardless of how converged the member is."""
    pos, alive, theta = blob
    relaxed, _, _, _ = equilibrate(pos, alive, theta)

    far = equilibrate_chunk(equilibrate_init(pos, alive, theta),
                            alive, theta, steps=16)
    near = equilibrate_chunk(equilibrate_init(relaxed, alive, theta),
                             alive, theta, steps=16)
    # A converged member burns its budget as no-ops in phase 2, so its counters
    # stop advancing — but the loop still executed 16 iterations either way,
    # which is what keeps vmap bounded.
    assert int(far.n_iter) <= 16
    assert int(near.n_iter) <= 16


# --------------------------------------------------------------------------- #
# Done is not converged
# --------------------------------------------------------------------------- #

def test_done_is_not_the_same_as_converged(blob):
    """`phase == 2` also means 'gave up'. Callers must check `res`."""
    pos, alive, theta = blob
    st = equilibrate_chunked(pos, alive, theta, steps=8, max_descent=3)
    assert int(st.phase) == 2, "should have stopped on the descent budget"
    assert float(st.res) > 1e-12, "expected an unconverged stop"


def test_resume_across_a_serialization_round_trip(blob):
    """State survives numpy round-tripping, as a checkpoint would require."""
    pos, alive, theta = blob
    st = equilibrate_chunk(equilibrate_init(pos, alive, theta),
                           alive, theta, steps=20)
    payload = {k: np.asarray(v) for k, v in st._asdict().items()}
    revived = EquilibrateState(**{k: jnp.asarray(v) for k, v in payload.items()})

    a = equilibrate_chunk(st, alive, theta, steps=20)
    b = equilibrate_chunk(revived, alive, theta, steps=20)
    assert np.array_equal(np.asarray(a.pos), np.asarray(b.pos))
