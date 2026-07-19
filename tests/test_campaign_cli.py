"""The campaign CLI's non-fleet surface: plan / estimate / local + the leg builder.

The `fleet` subcommand rents real hardware (`# pragma: no cover`); everything that
does not — spec loading, leg expansion, cost estimation, and an in-process `local`
run — is exercised here. Also covers the `evolve` RunFn's variation/selection seams
(retro, neutral) and the `evodevo_run` dispatch guard, which the smoke test doesn't
reach.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from jax_morpho import campaign as C
from jax_morpho.runs import RunConfig

# Tiny two-arm spec: a CPU mu_gate gate + a one-generation evolve lineage. Small
# enough to run in-process in a couple of seconds.
SPEC = {
    "gtag": "cli-test",
    "replicates": [0, 1],
    "arms": {
        "point": {
            "cfg": {"kind": "evolve", "n_pop": 20, "n_generations": 1,
                    "n_genes": 3, "dtype": "float64"},
            "knobs": {"selection": {"type": "truncation", "frac": 0.3},
                      "variation": {"type": "point", "rate": 0.03},
                      "n_rings": 1, "grn_hidden": 8, "n_loci_per_gene": 3,
                      "landmark_stride": 3, "sigma_env": 0.005},
        },
        "mu": {
            "cfg": {"kind": "mu_gate", "n_pop": 0, "n_generations": 0,
                    "n_genes": 0, "dtype": "float64"},
            "knobs": {"n_ind": 300, "n_replays": 3, "p2_values": [0.5, 0.03125]},
            "replicates": [0],          # a gate need not be swept
        },
    },
    "estimate": {"s_per_run": 12.0, "dph": 0.2, "failure_tax": 1.0,
                 "acq_tax_usd": 0.03},
}


def test_load_spec_default_and_file(tmp_path):
    assert C.load_spec(None) is C.DEFAULT_SPEC
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(SPEC))
    assert C.load_spec(str(p))["gtag"] == "cli-test"


def test_build_legs_axes_and_deterministic_seed():
    legs = C.build_legs(SPEC)
    assert len(legs) == 3                        # point×2 replicates + mu×1
    rids = {leg["rid"] for leg in legs}
    assert rids == {"point_r0", "point_r1", "mu_r0"}
    for leg in legs:                             # axes present for groupby
        assert "arm" in leg and "replicate" in leg
    # seed is a pure function of (arm, replicate)
    assert C._seed("point", 0) == C.build_legs(SPEC)[0]["seed"]
    assert C._seed("point", 0) != C._seed("point", 1)


def test_plan_configs_unique_hashes():
    configs = C.plan_configs(SPEC)
    assert len(configs) == 3
    assert all(isinstance(c, RunConfig) for c in configs)
    assert len({c.config_hash() for c in configs}) == 3     # no collisions
    # arm/replicate rode into params; seed did not
    c0 = configs[0]
    assert "arm" in c0.params and "seed" not in c0.params


def test_cmd_plan_via_main(capsys):
    assert C.main(["plan"]) == 0                  # default spec
    out = capsys.readouterr().out
    assert "legs:" in out and "kind=" in out


def test_cmd_estimate_reports_gpu_h_wall_and_usd(capsys):
    assert C.main(["estimate", "--hosts", "2"]) == 0
    out = capsys.readouterr().out
    assert "GPU-hours" in out and "wall-clock" in out and "cost (USD)" in out


def test_cmd_local_runs_and_reports(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    out_dir = tmp_path / "out"
    assert C.main(["--spec", str(spec), "local", "--out", str(out_dir)]) == 0
    report = capsys.readouterr().out
    # every planned run wrote a DONE.json...
    for c in C.plan_configs(SPEC):
        assert (out_dir / c.run_name() / "DONE.json").exists()
    # ...and the report surfaced both a lineage line and the M-U gate verdict
    assert "M-U gate" in report and ("PASS" in report or "FAIL" in report)


def test_report_results_handles_missing_done(tmp_path, capsys):
    """A run with no DONE.json is reported as such, not a crash."""
    configs = C.plan_configs(SPEC)
    C._report_results(str(tmp_path), configs)     # nothing on disk
    assert "(no result)" in capsys.readouterr().out


# -- runfns branches the smoke test doesn't reach ---------------------------

def _tiny_ctx():
    from run_farm.protocols import RunContext
    return RunContext(resume=None, resume_step=None, emit=lambda r: None,
                      checkpoint=lambda s, step: None,
                      trigger=lambda s, reason: None)


def test_evolve_retro_and_neutral_variation_run():
    """The §5c retro seam (needs a donor pool) and a neutral-drift lineage."""
    from jax_morpho.runfns import evolve_lineage
    knobs = {"n_rings": 1, "grn_hidden": 8, "n_loci_per_gene": 3,
             "landmark_stride": 3, "sigma_env": 0.005}
    retro = RunConfig(kind="evolve", n_pop=20, n_generations=1, n_genes=3, seed=1,
                      params={"arm": "retro",
                              "selection": {"type": "truncation", "frac": 0.3},
                              "variation": {"type": "retro", "rate": 1.0,
                                            "genes_per_event": 1, "n_donors": 4},
                              **knobs})
    r = evolve_lineage(retro, _tiny_ctx())
    assert r["kind"] == "evolve" and r["heterozygosity_final"] is not None

    neutral = RunConfig(kind="evolve", n_pop=20, n_generations=1, n_genes=3, seed=2,
                        params={"arm": "neutral", "develop": False,
                                "selection": {"type": "neutral", "frac": 1.0},
                                "variation": {"type": "point", "rate": 0.02},
                                "n_loci_per_gene": 3})
    rn = evolve_lineage(neutral, _tiny_ctx())
    assert rn["kind"] == "evolve"


def test_evodevo_run_rejects_unknown_kind():
    from jax_morpho.runfns import evodevo_run
    with pytest.raises(ValueError, match="unknown run kind"):
        evodevo_run(RunConfig(kind="nonsense"), _tiny_ctx())
