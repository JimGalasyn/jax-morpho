"""CPU–GPU equivalence of `equilibrate`: the same fixed point, up to a rigid motion.

The claim a backend comparison can make is *"the two backends land on the same fixed
point"* — not that they agree to the bit. Bit-identity is a same-process property
(`test_chunked_equilibrate` uses it across chunk schedules, where it belongs); across
backends and hosts the arithmetic differs at the ULP level by kernel selection and
reduction order, and a hardcoded digest is green only on the machine that recorded it.

What is compared
----------------
The energy is invariant under translation and rotation, so the equilibrium is a
3-parameter family and the solver's PATH picks the member. Raw coordinates therefore
test "the same trajectory", not "the same fixed point": two converged CPU runs that
differ only in `newton_tol` have the same shape to 2 ULP and raw coordinates 92–1020
floors apart. So every comparison here is of PAIRWISE DISTANCES (`_shape`), which fix
the gauge.

The tolerance
-------------
Per seed, the floor is the larger of

* `ULP_MARGIN` ULPs of the largest pairwise distance. This is the term IN FORCE: on the
  box that wrote this it is the floor in 12/12 seeds, by 29–57×; and
* the same-backend CODE-PATH difference: `equilibrate` (a `while_loop`) against
  `equilibrate_chunked` (a `fori_loop` over a `switch`) on the CPU. It can only loosen
  the floor, so the chunked leg is held to the same convergence and the same iteration
  count as the monolithic one — an unconverged leg would otherwise set a floor of 0.55,
  inside which any answer passes.

The tolerance is calibrated on both sides, on the CPU alone, so both halves run in CI:

* a NULL — a change of path (`newton_tol` 1e-4 → 1e-3) that reaches the same fixed point
  in 9–20 % fewer iterations must sit inside the floor. Measured: ≤ 0.04 floors; and
* a PLANTED EFFECT — a relative change of 1e-13 in `r_eq` must sit outside it. Measured:
  12 floors on every seed, on a response that is linear from 1e-14 to 1e-3.

Together they hold `ULP_MARGIN` between ~2 and ~100. A uniform rescale of `D` is the
mutation to keep in mind: it multiplies the energy by a constant, so it changes the path
and not the fixed point, and a planted-effect check that passes on it is measuring the
gauge.

Measured on the box that wrote this (RTX 4090 Laptop, jax 0.10.2, x64): 12 seeds, both
backends converged with identical iteration counts (172–359), pairwise distances agree
to 1.3e-15 or better (≤ 0.05 floors), code-path term 4.4e-16–8.9e-16.

Not claimed
-----------
* That each seed has its own equilibrium. With a uniform θ the 12 jittered scenes relax
  to ONE packing under six labellings: the jitter varies the path and the gauge. The
  comparison is over a set of trajectories.
* That this floor transfers to a heterogeneous θ. There the polish stops at a residual
  of ~1e-13 rather than at roundoff, and the same change of path moves the shape by up
  to 11 floors (5 % jitter of `D` and `r_eq`, 12 seeds); that case needs a floor scaled
  by the residual.
* That a backend whose trajectory forks into another basin is equivalent. It lands on a
  different fixed point and fails here; the report prints the iteration counts, which is
  where a fork shows first.

Skips LOUDLY where no GPU is visible (CI runs `JAX_PLATFORMS=cpu`), and fails instead
when `JAX_MORPHO_REQUIRE_GPU=1`, so a skip cannot read as a pass on a box that was
expected to have one.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_morpho.evodevo.mechanical import (
    equilibrate,
    equilibrate_chunked,
    hex_blob,
    pack_theta,
    uniform_theta,
    unpack_theta,
)

jax.config.update("jax_enable_x64", True)

SEEDS = range(12)
JITTER = 0.05          # cell-diameter fraction: varies the path and the gauge per seed
TOL = 1e-12
ULP_MARGIN = 32
PATH_NEWTON_TOL = 1e-3  # the null: hand over to Newton a decade early (default 1e-4)
PLANTED = 1e-13        # the effect: a relative change of r_eq, ~12 floors
PLANTED_MIN = 4.0      # floors; a third of the measured response


def _gpu():
    return [d for d in jax.devices() if d.platform == "gpu"]


def _require_gpu():
    devs = _gpu()
    if devs:
        return devs[0]
    if os.environ.get("JAX_MORPHO_REQUIRE_GPU", "").strip().lower() in {"1", "true", "yes"}:
        pytest.fail("JAX_MORPHO_REQUIRE_GPU is set and jax sees no GPU device")
    pytest.skip("no GPU device visible (CI runs JAX_PLATFORMS=cpu); a skip reads as a pass — "
                "set JAX_MORPHO_REQUIRE_GPU=1 on a box that is expected to have one")


def _cpu():
    try:
        return jax.devices("cpu")[0]
    except RuntimeError as e:
        err = e
    pytest.fail(f"jax has no CPU backend ({err}); every leg here is measured against the "
                "CPU — run with JAX_PLATFORMS=cuda,cpu, or leave it unset")


def _scene(seed: int):
    base = hex_blob(2)                                   # 19 cells
    rng = np.random.default_rng(seed)
    pos = base + JITTER * rng.standard_normal(base.shape)
    return pos, uniform_theta(pos.shape[0])


def _shape(x):
    """Pairwise distances: the fixed point modulo the rigid motion the path chose."""
    d = x[:, None, :] - x[None, :, :]
    return np.sqrt((d * d).sum(-1))


def _solve(dev, pos, theta, *, chunked: bool = False, r_eq_scale: float = 1.0, **kw):
    p = jax.device_put(jnp.asarray(pos, dtype=jnp.float64), dev)
    alive = jax.device_put(jnp.ones(p.shape[0], dtype=bool), dev)
    D, r_eq = unpack_theta(jnp.asarray(theta))
    th = jax.device_put(pack_theta(D, r_eq * r_eq_scale), dev)
    if chunked:
        st = equilibrate_chunked(p, alive, th, steps=64, tol=TOL, **kw)
        x, res, n = st.pos, st.res, st.n_iter
        conv = (int(st.phase) == 2) and (float(res) <= TOL)      # done is not converged
    else:
        x, res, n, conv = equilibrate(p, alive, th, tol=TOL, **kw)
    # np.asarray drops the device: a leg that fell back to the CPU would compare equal
    assert x.devices() == {dev}, f"asked for {dev}, ran on {x.devices()}"
    return np.asarray(x), float(res), int(n), bool(conv)


def _floor(cpu, pos, theta, xc, nc):
    """max(ULP_MARGIN ulp of the pairwise distances, same-backend code-path difference)."""
    xk, rk, nk, ck = _solve(cpu, pos, theta, chunked=True)
    assert ck and nk == nc, (f"chunked leg of the floor: converged={ck} residual={rk:.2e} "
                             f"iters {nk} against {nc}")
    sc = _shape(xc)
    code_path = float(np.abs(_shape(xk) - sc).max())
    ulp = ULP_MARGIN * np.finfo(np.float64).eps * float(sc.max())
    return max(code_path, ulp), code_path, ulp


def test_gpu_lands_on_the_cpu_fixed_point_within_the_floor():
    gpu = _require_gpu()
    cpu = _cpu()
    ratios, rows = [], []
    for seed in SEEDS:
        pos, theta = _scene(seed)
        xc, rc, nc, cc = _solve(cpu, pos, theta)
        xg, rg, ng, cg = _solve(gpu, pos, theta)
        assert cc and cg, f"seed {seed}: converged cpu={cc} gpu={cg}"
        assert rc <= TOL and rg <= TOL, f"seed {seed}: residual cpu={rc:.2e} gpu={rg:.2e}"
        floor, code_path, ulp = _floor(cpu, pos, theta, xc, nc)
        d = float(np.abs(_shape(xg) - _shape(xc)).max())
        ratios.append(d / floor)
        rows.append(f"seed {seed}: |gpu-cpu| {d:.2e}  floor {floor:.2e}  (code-path {code_path:.2e}, "
                    f"{ULP_MARGIN} ulp {ulp:.2e})  iters {nc}/{ng}")
    ratios = np.asarray(ratios)
    report = "\n".join(rows) + f"\nmedian d/floor {np.median(ratios):.3f}, max {ratios.max():.3f}"
    # every seed inside the floor: the two backends share the fixed point
    assert ratios.max() <= 1.0, report


def test_the_floor_passes_a_change_of_path_to_the_same_fixed_point():
    """The null, on the CPU alone: handing over to Newton a decade early changes the
    trajectory and not the energy, so it must land inside the floor. In raw coordinates
    it lands 92–1020 floors out, which is what comparing them measures."""
    cpu = _cpu()
    rows = []
    for seed in SEEDS:
        pos, theta = _scene(seed)
        xc, _, nc, cc = _solve(cpu, pos, theta)
        xn, _, nn, cn = _solve(cpu, pos, theta, newton_tol=PATH_NEWTON_TOL)
        assert cc and cn, f"seed {seed}: converged {cc}/{cn}"
        assert nn != nc, f"seed {seed}: {nn} iterations on both paths — the path did not change"
        floor, _, _ = _floor(cpu, pos, theta, xc, nc)
        rows.append(float(np.abs(_shape(xn) - _shape(xc)).max()) / floor)
    assert max(rows) <= 1.0, f"a change of path moves the shape by {max(rows):.2f} floors: {rows}"


def test_the_floor_discriminates_a_planted_effect():
    """The effect, on the CPU alone: a relative change of 1e-13 in `r_eq` is a different
    fixed point and must land outside the floor, on every seed. If it did not, the
    equivalence test above could pass a backend that lands somewhere else."""
    cpu = _cpu()
    rows = []
    for seed in SEEDS:
        pos, theta = _scene(seed)
        xc, _, nc, cc = _solve(cpu, pos, theta)
        xp, _, _, cp = _solve(cpu, pos, theta, r_eq_scale=1.0 + PLANTED)
        assert cc and cp, f"seed {seed}: converged {cc}/{cp}"
        floor, _, _ = _floor(cpu, pos, theta, xc, nc)
        rows.append(float(np.abs(_shape(xp) - _shape(xc)).max()) / floor)
    assert min(rows) >= PLANTED_MIN, (f"a relative change of {PLANTED:.0e} in r_eq moves the shape "
                                      f"by only {min(rows):.2f} floors: {rows}")
