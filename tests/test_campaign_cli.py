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


# --- fleet wiring: productized RunPod-sshd onstart + fast-fail executor --------
# `cmd_fleet` rents hardware (`# pragma: no cover`), but the wiring it delegates to
# — onstart builders, key reading, the fast-fail readiness split, and provider→onstart
# selection in `_build_fleet` — is pure and exercised here (providers monkeypatched).

def _fleet_args(**over):
    import argparse
    base = dict(provider="vast", cap_usd=5.0, gpu="RTX_3090", max_dph=0.40,
                cloud_type="secure", ssh_key="~/.ssh/vastai", ssh_grace=150.0,
                ready_timeout=480.0, rent_timeout=240.0, max_attempts=20,
                image="python:3.11-slim", ledger="x/ledger.jsonl", out="x")
    base.update(over)
    return argparse.Namespace(**base)


def test_vast_onstart_installs_engine_no_sshd():
    o = C._vast_onstart()
    assert "jax-morpho @" in o and "archive/refs/heads/main.tar.gz" in o
    assert "ENGINE_READY" in o
    # Vast injects sshd via runtype=ssh; no sshd/apt in the onstart (the tarball
    # install is exactly what let us drop apt-get git and its slow-mirror stall).
    assert "sshd" not in o and "apt-get" not in o


def test_runpod_onstart_starts_sshd_authorizes_key_and_holds_open():
    pub = "ssh-ed25519 AAAAUNIQUEKEY42 user@host"
    o = C._runpod_onstart(pub)
    assert "openssh-server" in o and "/usr/sbin/sshd" in o
    assert f"'{pub}'" in o                              # authorized, shell-quoted
    assert "jax-morpho @" in o and "ENGINE_READY" in o
    assert o.rstrip().endswith("sleep infinity")        # keep container + sshd alive


def test_runpod_onstart_shell_quotes_key_safely():
    # A pathological "key" with shell metachars must not break out of the echo.
    o = C._runpod_onstart("evil'; rm -rf / #")
    assert "rm -rf /" in o                               # present only inside the quote
    assert "'evil'\"'\"'; rm -rf / #'" in o              # shlex.quote form


def test_read_pubkey(tmp_path):
    key = tmp_path / "id"
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAAKEY x\n")
    assert C._read_pubkey(str(key)) == "ssh-ed25519 AAAAKEY x"
    with pytest.raises(FileNotFoundError):
        C._read_pubkey(str(tmp_path / "missing"))


class _DummyHost:
    id = "h1"
    ssh_host = "1.2.3.4"
    ssh_port = 22


def _fast_fail(**kw):
    from run_farm import LaunchSpec
    launch = LaunchSpec(image="python:3.11-slim", onstart="x", label="t")
    return C.FastFailExecutor(object(), "jax_morpho.runfns:evodevo_run", launch,
                              config_class=C.CONFIG_CLASS_REF, **kw)


def test_fast_fail_ssh_dead_fails_over_within_grace(monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda *_: None)
    monkeypatch.setattr(C, "_ssh", lambda *a, **k: (255, "connect refused"))
    ex = _fast_fail(ssh_grace=0.05, ready_timeout=5.0)
    with pytest.raises(C.HostProbeFailed, match="SSH never reachable"):
        ex._wait_engine_ready(_DummyHost())


def test_fast_fail_engine_install_times_out(monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda *_: None)

    def fake_ssh(key, host, port, cmd, timeout=30):     # SSH up, import never succeeds
        return (0, "OK") if "echo OK" in cmd else (1, "ModuleNotFoundError")

    monkeypatch.setattr(C, "_ssh", fake_ssh)
    ex = _fast_fail(ssh_grace=5.0, ready_timeout=0.05)
    with pytest.raises(C.HostProbeFailed, match="engine not ready"):
        ex._wait_engine_ready(_DummyHost())


def test_fast_fail_returns_when_engine_imports(monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda *_: None)
    monkeypatch.setattr(C, "_ssh", lambda *a, **k: (0, "OK"))    # everything succeeds
    ex = _fast_fail(ssh_grace=5.0, ready_timeout=5.0)
    assert ex._wait_engine_ready(_DummyHost()) is None          # no raise = ready


def _patch_providers(monkeypatch, recorded):
    import run_farm

    class FakeRunPod:
        def __init__(self, *a, ledger=None, cloud_type=None, interruptible=None, **k):
            recorded["cloud_type"] = cloud_type
            recorded["interruptible"] = interruptible

    class FakeVast:
        def __init__(self, *a, ledger=None, **k):
            recorded["vast"] = True

    monkeypatch.setattr(run_farm, "RentalLedger", lambda *a, **k: object())
    monkeypatch.setattr(run_farm, "RunPodProvider", FakeRunPod)
    monkeypatch.setattr(run_farm, "VastProvider", FakeVast)
    monkeypatch.setattr(run_farm, "CappedProvider", lambda base, cap, ledger: base)


def test_build_fleet_runpod_uses_sshd_onstart_and_secure(tmp_path, monkeypatch):
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAAKEY x\n")
    rec = {}
    _patch_providers(monkeypatch, rec)
    args = _fleet_args(provider="runpod", ssh_key=str(tmp_path / "id"),
                       out=str(tmp_path))
    _prov, launch, executor, configs = C._build_fleet(SPEC, args)
    assert "/usr/sbin/sshd" in launch.onstart            # sshd bootstrap chosen
    assert "'ssh-ed25519 AAAAKEY x'" in launch.onstart   # our key authorized
    assert rec["cloud_type"] == "SECURE" and rec["interruptible"] is False
    assert isinstance(executor, C.FastFailExecutor)
    assert executor.remote_work_dir == C._REMOTE_WORK_DIR
    assert executor.ssh_grace == args.ssh_grace
    assert len(configs) == 3                             # SPEC: point×2 + mu×1


def test_build_fleet_vast_uses_plain_onstart(tmp_path, monkeypatch):
    rec = {}
    _patch_providers(monkeypatch, rec)
    args = _fleet_args(provider="vast", out=str(tmp_path))
    _prov, launch, executor, _configs = C._build_fleet(SPEC, args)
    assert "/usr/sbin/sshd" not in launch.onstart        # Vast injects sshd itself
    assert "jax-morpho @" in launch.onstart
    assert rec.get("vast") is True


def test_build_fleet_rejects_unknown_provider(tmp_path, monkeypatch):
    rec = {}
    _patch_providers(monkeypatch, rec)
    args = _fleet_args(provider="gcp", out=str(tmp_path))
    with pytest.raises(ValueError, match="unknown provider"):
        C._build_fleet(SPEC, args)


def test_fast_fail_clamps_probe_timeout_to_remaining_budget(monkeypatch):
    # Regression: a probe's wall-timeout must be clamped to the time left, so a
    # dead host fails over within the budget instead of budget + a full 15s/30s probe.
    monkeypatch.setattr(C.time, "sleep", lambda *_: None)
    seen = []

    def rec_ssh(key, host, port, cmd, timeout=30):
        seen.append(timeout)
        return (255, "refused")

    monkeypatch.setattr(C, "_ssh", rec_ssh)
    ex = _fast_fail(ssh_grace=0.05, ready_timeout=5.0)
    with pytest.raises(C.HostProbeFailed, match="SSH never reachable"):
        ex._wait_engine_ready(_DummyHost())
    assert seen and all(t <= 0.05 + 1e-9 for t in seen)   # never the 15s probe cap
