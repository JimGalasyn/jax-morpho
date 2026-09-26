"""CPU–GPU equivalence of `equilibrate`, in distribution, against a measured floor.

The claim a backend comparison can make is *"the two backends land on the same fixed
point"* — not that they agree to the bit. Bit-identity is a same-process property
(`test_chunked_equilibrate` uses it across chunk schedules, where it belongs); across
backends and hosts the arithmetic differs at the ULP level by kernel selection and
reduction order, and a hardcoded digest is green only on the machine that recorded it.
So this test is the statistical form: over a set of seeds, the per-seed disagreement
between backends must sit inside a FLOOR measured on one backend, and the floor must be
shown to discriminate by a planted effect.

The floor, per seed, is the larger of

* the same-backend CODE-PATH difference: `equilibrate` (a `while_loop`) against
  `equilibrate_chunked` (a `fori_loop` over a `switch`) on the CPU — the solver's own
  answer to "how much does the fixed point move when only the arithmetic order changes";
  `test_chunked_equilibrate` measured it at one ULP; and
* `ULP_MARGIN` ULPs of the largest coordinate — reductions on a GPU reorder sums, and a
  few ULP per reduction is the expected size. 32 keeps a margin of ~10^10 against a
  genuinely different fixed point, which the planted-effect check below quantifies.

Measured on the box that wrote this (RTX 4090 Laptop, jax 0.10.2, x64): 12 seeds, both
backends converged with identical iteration counts, |gpu − cpu| 2–9e-16 (1–4 ULP),
code-path floor 1–7e-16.

Skips LOUDLY where no GPU is visible (CI runs `JAX_PLATFORMS=cpu`), and fails instead
when `JAX_MORPHO_REQUIRE_GPU` is set, so a skip cannot read as a pass on a box that was
expected to have one. The seeds jitter a 19-cell hex blob so that no two scenes share
a fixed point and the comparison is over a set, not a single trajectory.
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
    uniform_theta,
)

jax.config.update("jax_enable_x64", True)

SEEDS = range(12)
JITTER = 0.05          # cell-diameter fraction: enough to give every seed its own fixed point
TOL = 1e-12
ULP_MARGIN = 32
PLANTED = 0.01         # a 1 % change of θ: a different fixed point, by a distance the floor
                       # must be far below (else the floor is vacuous)


def _gpu():
    return [d for d in jax.devices() if d.platform == "gpu"]


def _require_gpu():
    devs = _gpu()
    if devs:
        return devs[0]
    if os.environ.get("JAX_MORPHO_REQUIRE_GPU"):
        pytest.fail("JAX_MORPHO_REQUIRE_GPU is set and jax sees no GPU device")
    pytest.skip("no GPU device visible (CI runs JAX_PLATFORMS=cpu); a skip reads as a pass — "
                "set JAX_MORPHO_REQUIRE_GPU=1 on a box that is expected to have one")


def _scene(seed: int):
    base = hex_blob(2)                                   # 19 cells
    rng = np.random.default_rng(seed)
    pos = base + JITTER * rng.standard_normal(base.shape)
    return pos, uniform_theta(pos.shape[0])


def _solve(dev, pos, theta, *, chunked: bool = False, theta_scale: float = 1.0):
    p = jax.device_put(jnp.asarray(pos, dtype=jnp.float64), dev)
    alive = jax.device_put(jnp.ones(p.shape[0], dtype=bool), dev)
    th = jax.device_put(jnp.asarray(theta) * theta_scale, dev)
    if chunked:
        st = equilibrate_chunked(p, alive, th, steps=64)
        return np.asarray(st.pos), None, int(st.n_iter), None
    x, res, n, conv = equilibrate(p, alive, th, tol=TOL)
    return np.asarray(x), float(res), int(n), bool(conv)


def _floor(cpu, pos, theta, xc):
    """max(same-backend code-path difference, ULP_MARGIN ulp of the coordinates)."""
    xk, _, _, _ = _solve(cpu, pos, theta, chunked=True)
    code_path = float(np.abs(xk - xc).max())
    ulp = ULP_MARGIN * np.finfo(np.float64).eps * float(np.abs(xc).max())
    return max(code_path, ulp), code_path, ulp


def test_gpu_lands_on_the_cpu_fixed_point_within_the_measured_floor():
    gpu = _require_gpu()
    cpu = jax.devices("cpu")[0]
    ratios, rows = [], []
    for seed in SEEDS:
        pos, theta = _scene(seed)
        xc, rc, nc, cc = _solve(cpu, pos, theta)
        xg, rg, ng, cg = _solve(gpu, pos, theta)
        assert cc and cg, f"seed {seed}: converged cpu={cc} gpu={cg}"
        assert rc <= TOL and rg <= TOL, f"seed {seed}: residual cpu={rc:.2e} gpu={rg:.2e}"
        floor, code_path, ulp = _floor(cpu, pos, theta, xc)
        d = float(np.abs(xg - xc).max())
        ratios.append(d / floor)
        rows.append(f"seed {seed}: |gpu-cpu| {d:.2e}  floor {floor:.2e}  (code-path {code_path:.2e}, "
                    f"{ULP_MARGIN} ulp {ulp:.2e})  iters {nc}/{ng}")
    ratios = np.asarray(ratios)
    report = "\n".join(rows) + f"\nmedian d/floor {np.median(ratios):.3f}, max {ratios.max():.3f}"
    # every seed inside the floor: the two backends share the fixed point
    assert ratios.max() <= 1.0, report


def test_the_floor_discriminates_a_planted_effect():
    """The tolerance is calibrated on a planted effect, not only on the null: a 1 % change
    of θ on the CPU alone moves the fixed point by ≥ 10^6 floors. If it did not, the
    equivalence test above could pass a backend that lands somewhere else. Runs on the
    CPU only, so it runs in CI."""
    cpu = jax.devices("cpu")[0]
    worst = np.inf
    for seed in list(SEEDS)[:4]:
        pos, theta = _scene(seed)
        xc, _, _, cc = _solve(cpu, pos, theta)
        xp, _, _, cp = _solve(cpu, pos, theta, theta_scale=1.0 + PLANTED)
        assert cc and cp
        floor, _, _ = _floor(cpu, pos, theta, xc)
        worst = min(worst, float(np.abs(xp - xc).max()) / floor)
    assert worst >= 1e6, f"planted 1 % θ moves the fixed point by only {worst:.1e} floors"
