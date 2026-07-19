"""The campaign seam exercised end to end over a LocalExecutor — no cloud, no cost.

Drives the full A/B/C contract locally over the morpho RunFn: register each config,
checkpoint per generation, stream events, finish — plus the two properties the whole
run-farm dependency exists to buy:

  * **idempotent skip (mechanism A):** re-submitting a finished campaign is a no-op,
    which is what makes spot preemption free.
  * **restart-exactness (contract B):** a lineage resumed from a mid-run checkpoint
    reproduces the uninterrupted final population bit-for-bit.

And the arm that makes the campaign self-validating: the Milocco–Uller Fig 3C
known-answer gate (DESIGN.md §5b).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)   # evodevo runs in x64; set before arrays

import json

import numpy as np
from run_farm import (FileRunRegistry, JsonlEventSink, LocalExecutor,
                      ProbeAdmission, run_campaign)
from run_farm.protocols import RunContext

from jax_morpho.runfns import evodevo_run
from jax_morpho.runs import RunConfig

# Deliberately tiny — a couple of small lineages and a coarse gate — so the whole
# contract runs in a few seconds on a laptop CPU.
_KNOBS = {"n_rings": 1, "grn_hidden": 8, "n_loci_per_gene": 3,
          "landmark_stride": 3, "sigma_env": 0.005}


def _evolve_cfg(arm, rep, n_gen=2, n_pop=24):
    return RunConfig(
        kind="evolve", n_pop=n_pop, n_generations=n_gen, n_genes=3, seed=rep + 1,
        params={"arm": arm, "replicate": rep,
                "selection": {"type": "truncation", "frac": 0.3},
                "variation": {"type": "point", "rate": 0.03}, **_KNOBS})


def _mu_cfg():
    return RunConfig(kind="mu_gate", seed=0,
                     params={"arm": "mu_reference", "n_ind": 400, "n_replays": 4,
                             "p2_values": [0.5, 0.03125, 0.001953125]})


def _run(configs, out):
    registry = FileRunRegistry(out)
    sink = JsonlEventSink()
    run_campaign(configs, evodevo_run, registry=registry,
                 sink=sink, admission=ProbeAdmission(require_gpu=False),
                 executor=LocalExecutor())


def test_local_campaign_writes_hashed_runs_and_is_idempotent(tmp_path):
    out = str(tmp_path / "campaign_out")
    configs = [_evolve_cfg("point", 0), _mu_cfg()]

    _run(configs, out)

    # Each config wrote its hashed dir + a DONE.json result; MANIFEST has both.
    for c in configs:
        assert (tmp_path / "campaign_out" / c.run_name() / "DONE.json").exists()
    manifest = (tmp_path / "campaign_out" / "MANIFEST.jsonl").read_text()
    assert all(c.run_name() in manifest for c in configs)

    # A re-run is the idempotent skip: every run already complete, nothing reruns.
    done = tmp_path / "campaign_out" / configs[0].run_name() / "DONE.json"
    before = done.read_text()
    _run(configs, out)                       # mechanism A: no-op
    assert done.read_text() == before


def test_mu_gate_arm_passes_its_known_answer(tmp_path):
    """The developmental G reproduces Fig 3C: G tracks the recombinant response
    (small angle), P does not (larger angle). This is the validation that travels
    with every campaign instead of being cited from the paper."""
    out = str(tmp_path / "campaign_out")
    cfg = _mu_cfg()
    _run([cfg], out)
    # finish() writes the RunFn's returned result dict straight to DONE.json.
    result = json.loads((tmp_path / "campaign_out" / cfg.run_name()
                         / "DONE.json").read_text())
    assert result["kind"] == "mu_gate"
    assert result["gate_pass"] is True
    assert result["worst_angle_G"] < result["median_angle_P"]


def test_evolve_lineage_restart_is_exact():
    """A lineage resumed from its gen-1 checkpoint reaches the identical final
    population as the uninterrupted run — the property that makes a preempted
    $-campaign resume instead of restart-from-zero."""
    cfg = _evolve_cfg("point", 3, n_gen=3, n_pop=30)

    # Full run, capturing every checkpoint and the final triggered capture.
    ckpts, trig = {}, {}
    ctx = RunContext(
        resume=None, resume_step=None, emit=lambda r: None,
        checkpoint=lambda s, step: ckpts.__setitem__(
            step, {k: np.asarray(v).copy() for k, v in s.items()}),
        trigger=lambda s, reason: trig.update(
            {k: np.asarray(v).copy() for k, v in s.items()}))
    full = evodevo_run(cfg, ctx)

    # Resume from the gen-1 checkpoint and finish.
    ckpts2 = {}
    trig2 = {}
    ctx2 = RunContext(
        resume=ckpts[1], resume_step=1, emit=lambda r: None,
        checkpoint=lambda s, step: ckpts2.__setitem__(step, s),
        trigger=lambda s, reason: trig2.update(
            {k: np.asarray(v).copy() for k, v in s.items()}))
    resumed = evodevo_run(cfg, ctx2)

    assert resumed["heterozygosity_final"] == full["heterozygosity_final"]
    assert np.array_equal(trig2["scores"], trig["scores"])
