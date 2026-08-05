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


# ------------------------------------------------------------ remote env ----
# Variables the worker must see AT PROCESS START. onstart cannot supply them (the
# worker arrives over a non-interactive `ssh host cmd`, which sources no profile),
# and omitting them is silent: legs run, numbers look fine, only the resume-exactness
# claim is void.

def test_remote_env_defaults_to_pinning_autotune():
    """A campaign that checkpoints promises resume exactness, so the safe value is
    the default rather than an opt-in someone has to know to ask for."""
    assert C.resolve_remote_env(None) == {"XLA_FLAGS": "--xla_gpu_autotune_level=0"}


def test_remote_env_none_ships_nothing():
    """'I want the old behaviour' must be sayable on the command line, and visible
    there, rather than requiring an edit to the source."""
    assert C.resolve_remote_env(["none"]) == {}
    assert C.resolve_remote_env(["NONE"]) == {}


def test_remote_env_keeps_values_containing_equals_and_spaces():
    """XLA_FLAGS is a space-separated list of --k=v; splitting on every '=' would
    silently truncate it to the first flag."""
    assert C.resolve_remote_env(["A=1", "XLA_FLAGS=--x=0 --y=1"]) == {
        "A": "1", "XLA_FLAGS": "--x=0 --y=1"}


@pytest.mark.parametrize("bad", ["novalue", "=1"])
def test_remote_env_rejects_malformed(bad):
    with pytest.raises(ValueError):
        C.resolve_remote_env([bad])


# ------------------------------------------------------------ provenance ----
def test_every_result_records_its_floating_point_environment():
    """A number without its backend is not reproducible in practice.

    This whole mechanism exists because the same lineage gives different answers in
    different processes when XLA autotuning is on, and CPU digests differ between
    machines at identical library versions. A result whose backend and XLA flags were
    never recorded cannot be compared with a later one except by assuming they match.
    """
    from jax_morpho import runfns

    env = runfns._provenance()
    assert set(env) == {"backend", "device", "jax_version", "xla_flags"}
    assert env["backend"] in {"cpu", "gpu", "tpu", "unknown"}
    assert env["jax_version"]


def test_provenance_reports_the_xla_flags_actually_in_force(monkeypatch):
    """The farm sets these via remote_env; the record is how we know they arrived."""
    from jax_morpho import runfns

    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    assert runfns._provenance()["xla_flags"] == "--xla_gpu_autotune_level=0"
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    assert runfns._provenance()["xla_flags"] == ""


def test_provenance_never_overwrites_a_measurement():
    """Nested under 'env' precisely so a future result key named 'backend' cannot
    silently clobber, or be clobbered by, provenance."""
    from jax_morpho import runfns

    cfg = RunConfig(kind="mu_gate", n_genes=3, seed=0,
                    params={"arm": "g", "n_points": 3})

    class _Ctx:
        resume = None
        resume_step = None

        def emit(self, row): pass
        def checkpoint(self, state, step): pass
        def trigger(self, payload, reason): pass

    out = runfns.evodevo_run(cfg, _Ctx())
    assert out["kind"] == "mu_gate"              # the measurement survives
    assert "env" in out and out["env"]["backend"]


# ------------------------------------------------------------------ cuda ----
def test_cuda_install_precedes_the_engine():
    """Order is the whole point. jax-morpho depends on plain `jax`, so installing the
    engine first pulls the CPU wheel and the box then computes on CPU while looking,
    from the driver's side, exactly like a successful GPU campaign."""
    block = C._install_block(cuda=True)
    assert block.index("jax[cuda12]") < block.index("jax-morpho @")


def test_no_cuda_flag_leaves_the_bootstrap_unchanged():
    block = C._install_block(cuda=False)
    assert "cuda12" not in block
    # Same engine install, just invoked through an explicit interpreter.
    assert block.endswith(C._INSTALL.replace("pip install", "-m pip install", 1)
                          .lstrip())
    assert block.count("pip install") == 1


@pytest.mark.parametrize("builder,args", [
    (lambda cuda: C._vast_onstart(cuda=cuda), ()),
    (lambda cuda: C._runpod_onstart("ssh-ed25519 AAAA test", cuda=cuda), ()),
])
def test_both_providers_honour_cuda(builder, args):
    assert "jax[cuda12]" in builder(True)
    assert "jax[cuda12]" not in builder(False)
    # Readiness marker must survive either way, or the probe never sees the box up.
    assert "ENGINE_READY" in builder(True)


# -------------------------------------------------- install/run interpreter ----
def test_install_uses_the_same_interpreter_the_worker_will_run():
    """Never bare `pip`. Two live failures on 2026-08-04 forced this.

    On pytorch/pytorch:2.5.1-cuda12.4 the interpreter lives in /opt/conda/bin, which
    a non-interactive `bash -c` does not have on PATH, so `pip install` was
    `command not found`; `set -e` then killed the onstart AFTER sshd had started,
    leaving an SSH-reachable box that could never become ready and burned the whole
    ready_timeout. And even where `pip` exists it may belong to a different
    interpreter than the worker's, installing the engine where it is never found.
    """
    block = C._install_block(cuda=True, python="/opt/conda/bin/python")
    assert "pip install" in block
    assert block.count("/opt/conda/bin/python -m pip install") == 2   # cuda + engine
    # A bare `pip install` must not survive anywhere in the block.
    for line in block.splitlines():
        assert not line.strip().startswith("pip "), line


def test_onstart_installs_into_the_configured_interpreter():
    for build in (lambda: C._vast_onstart(cuda=True, python="/opt/conda/bin/python"),
                  lambda: C._runpod_onstart("k", cuda=True,
                                            python="/opt/conda/bin/python")):
        s = build()
        assert "/opt/conda/bin/python -m pip install" in s
        assert "\npip install" not in s


def test_default_interpreter_is_still_plain_python():
    """python:3.11-slim (the default image) has `python` on PATH; don't regress it."""
    assert "python -m pip install" in C._install_block(cuda=False)


# ------------------------------------------------ unreleased-dep import gate ----
def test_campaign_imports_without_the_unreleased_gauntlet_check(monkeypatch):
    """`import jax_morpho.campaign` must work against a PUBLISHED run-farm.

    `RemoteEnvPinned` landed after run-farm 0.2.0 and is in no release yet. It was
    imported at module scope, so importing this module — and therefore collecting
    this whole test file — died with ImportError on CI, which installs run-farm from
    PyPI. It passed locally only because run-farm is installed EDITABLE from a
    checkout: the environment differed from the target in precisely the dependency
    that mattered.

    Simulated by hiding the symbol, so this fails again if the import moves back to
    module scope.
    """
    import importlib

    import run_farm.gauntlet as g

    monkeypatch.delattr(g, "RemoteEnvPinned", raising=False)
    mod = importlib.reload(importlib.import_module("jax_morpho.campaign"))
    assert mod.resolve_remote_env(None)          # module usable without the symbol
    importlib.reload(mod)                        # restore for the rest of the session


def test_the_env_guard_raises_rather_than_silently_skipping(monkeypatch):
    """A guard that quietly does not run is worse than no guard.

    What it guards — a missing worker env — is itself silent: every leg still runs
    and produces plausible numbers, and only the reproducibility claim is void. So a
    too-old run-farm must be an error, not a shrug.
    """
    import builtins

    real = builtins.__import__

    def no_gauntlet(name, *a, **kw):
        if name == "run_farm.gauntlet":
            raise ImportError("no RemoteEnvPinned in run-farm 0.2.0")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_gauntlet)
    with pytest.raises(RuntimeError, match="run-farm > 0.2.0"):
        C._load_env_guard()
