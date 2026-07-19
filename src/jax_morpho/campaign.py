"""`jax-morpho-campaign` — plan, cost, and launch an evodevo fleet campaign.

The governance entry point DESIGN.md §5b asks for: **cost is a first-class,
pre-launch quantity, and the dollar cap is enforced, not advisory.** The layered
work (understand → design → run) crosses the run-farm boundary here; every heavy
mechanism (config-hashed identity, restart-exact checkpoints, probe-or-bail
admission, teardown-verifying brokers, the enforced cap) is run-farm's — this file
only assembles them around the morpho RunFn.

Subcommands::

    jax-morpho-campaign plan      [--spec s.json]                  # legs + hashes
    jax-morpho-campaign estimate  [--spec s.json] [--hosts N]      # GPU-h, wall-h, $
    jax-morpho-campaign local     [--spec s.json] [--out DIR]      # run in-process
    jax-morpho-campaign fleet     --provider vast|runpod --cap-usd C  [...]

`plan`, `estimate`, and `local` are free and need no cloud. `fleet` rents real,
billable hardware behind a `CappedProvider` — it REFUSES to rent past `--cap-usd`
(booked + in-flight burn), so a slip cannot run the budget away.

Every campaign carries the Milocco–Uller Fig 3C known-answer arm (``kind:
"mu_gate"``): its result asserts the developmental G still reproduces their result,
so the validation ladder travels with the science instead of being cited from a
paper.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
import time
from pathlib import Path

# x64 before any array work: the implicit solve + Procrustes readout need it, and
# an in-process (local) run must set it itself — a remote worker gets it from
# run_farm.remote via the config dtype. Import-time is the only safe moment.
import jax

jax.config.update("jax_enable_x64", True)

from run_farm import (
    FileRunRegistry,
    JsonlEventSink,
    LocalExecutor,
    ProbeAdmission,
    ProviderExecutor,
    estimate,
    run_campaign,
)
# `_ssh` is the same host-probe helper the base ProviderExecutor uses (shared so
# our fast-fail readiness check keeps run-farm's SSH keepalive/timeout options).
from run_farm.provider_exec import _ssh
from run_farm.protocols import HostProbeFailed

from jax_morpho.farm_config import morpho_leg_to_config
from jax_morpho.runfns import evodevo_run

# The RunFn the worker imports by name on a rented box (never shipped as a closure).
RUN_FN_REF = "jax_morpho.runfns:evodevo_run"
CONFIG_CLASS_REF = "jax_morpho.runs:RunConfig"


# A small, self-contained default campaign: a gradualism-vs-punctuation contrast
# (§5c) across replicate lineages, plus the M-U known-answer gate. Override with
# --spec. Deliberately cheap so `local` runs on a laptop in a couple of minutes.
DEFAULT_SPEC = {
    "gtag": "morpho-default-campaign",
    "replicates": [0, 1, 2],
    "arms": {
        "point": {
            "cfg": {"kind": "evolve", "n_pop": 120, "n_generations": 8,
                    "n_genes": 4, "dtype": "float64"},
            "knobs": {"selection": {"type": "truncation", "frac": 0.2},
                      "variation": {"type": "point", "rate": 0.02},
                      "sigma_env": 0.005, "measure_response": True},
        },
        "retro": {
            "cfg": {"kind": "evolve", "n_pop": 120, "n_generations": 8,
                    "n_genes": 4, "dtype": "float64"},
            "knobs": {"selection": {"type": "truncation", "frac": 0.2},
                      "variation": {"type": "retro", "rate": 1.0,
                                    "genes_per_event": 1, "n_donors": 5},
                      "sigma_env": 0.005, "measure_response": True},
        },
        # The known-answer arm: one leg (replicate 0 only), not swept.
        "mu_gate": {
            "cfg": {"kind": "mu_gate", "n_pop": 0, "n_generations": 0,
                    "n_genes": 0, "dtype": "float64"},
            "knobs": {"n_ind": 1500, "n_replays": 12},
            "replicates": [0],
        },
    },
    # Cost model: measure these from your own RentalLedger; the defaults are the
    # 2026-07-16 live-survey figures (§2E: ~42% of created instances ever boot).
    "estimate": {"s_per_run": 90.0, "dph": 0.30, "failure_tax": 1.38,
                 "acq_tax_usd": 0.03},
}


def load_spec(path: str | None) -> dict:
    if path is None:
        return DEFAULT_SPEC
    return json.loads(Path(path).read_text())


def _seed(arm: str, rep: int) -> int:
    """A deterministic, collision-resistant per-leg seed. Caller-owned (run-farm
    never invents seeds); a stable scheme so a re-launch resumes identically."""
    import hashlib
    h = hashlib.sha256(f"{arm}:{rep}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def build_legs(spec: dict) -> list[dict]:
    """Expand (arm × replicate) into run-farm leg dicts.

    Each leg carries its axes explicitly (`arm`, `replicate`) so a downstream
    aggregation can groupby them off the result records, and its developmental
    knobs as top-level keys so `run_farm.farm.leg_params` lifts them into
    `params` (identity fields stay in `cfg`). The M-U arm can pin its own
    `replicates` (default: just replicate 0 — a gate need not be swept).
    """
    default_reps = spec.get("replicates", [0])
    legs: list[dict] = []
    for arm, adef in spec["arms"].items():
        reps = adef.get("replicates", default_reps)
        for rep in reps:
            legs.append({
                "rid": f"{arm}_r{rep}",
                "arm": arm,
                "replicate": rep,
                "seed": _seed(arm, rep),
                "cfg": dict(adef["cfg"]),
                # deep-copy: sibling replicates of one arm must not share nested
                # knob dicts (leg_params only copies cfg/plan/required_shas).
                **copy.deepcopy(adef.get("knobs", {})),   # -> top-level params

            })
    return legs


def plan_configs(spec: dict) -> list:
    """Legs -> `jax_morpho.RunConfig`s (the config_factory the fleet plans with)."""
    gtag = spec.get("gtag", "morpho-campaign")
    return [morpho_leg_to_config(leg, gtag, {}) for leg in build_legs(spec)]


# -- subcommands -------------------------------------------------------------


def cmd_plan(spec: dict, args) -> int:
    configs = plan_configs(spec)
    print(f"campaign '{spec.get('gtag')}' — {len(configs)} legs:")
    for c in configs:
        arm = c.params.get("arm")
        rep = c.params.get("replicate")
        print(f"  {c.run_name():<48}  arm={arm:<8} rep={rep}  kind={c.kind}")
    return 0


def cmd_estimate(spec: dict, args) -> int:
    e = dict(spec.get("estimate", {}))
    n_runs = len(build_legs(spec))
    n_hosts = args.hosts
    est = estimate(
        n_runs=n_runs,
        s_per_run=float(e.get("s_per_run", 90.0)),
        dph=float(e.get("dph", 0.30)),
        n_hosts=n_hosts,
        failure_tax=float(e.get("failure_tax", 0.0)),
        acq_tax_usd=float(e.get("acq_tax_usd", 0.0)),
    )
    # §5b: report GPU-h AND elapsed. A GPU-h-only price hides an infeasible
    # schedule (the marketplace is ~251 usable GPUs total, not a fleet-week).
    print(f"campaign '{spec.get('gtag')}' — {n_runs} runs over {n_hosts} host(s):")
    print(f"  GPU-hours (incl. failure tax) : {est['gpu_h']:>10.2f}")
    print(f"  wall-clock hours (even split) : {est['wall_h']:>10.2f}")
    print(f"  estimated cost (USD)          : {est['usd']:>10.2f}")
    print(f"  → per-run mean                : ${est['usd'] / max(1, n_runs):.4f}")
    return 0


def cmd_local(spec: dict, args) -> int:
    out = args.out
    registry = FileRunRegistry(out)
    sink = JsonlEventSink()
    admission = ProbeAdmission(require_gpu=False)   # laptop, not a fleet
    executor = LocalExecutor()
    configs = plan_configs(spec)
    run_campaign(configs, evodevo_run, registry=registry, sink=sink,
                 admission=admission, executor=executor)
    print(f"ran {len(configs)} config-hashed run dirs + MANIFEST under {out}/")
    _report_results(out, configs)
    return 0


def _read_pubkey(ssh_key: str) -> str:
    """The public half of `ssh_key` (`<ssh_key>.pub`), for a provider that must be
    told which key to authorize (RunPod). Raises with a clear message if absent."""
    pub = Path(os.path.expanduser(ssh_key + ".pub"))
    if not pub.exists():
        raise FileNotFoundError(
            f"public key {pub} not found — need it to authorize SSH on the rented "
            f"pod; generate the pair or pass --ssh-key")
    return pub.read_text().strip()


def _build_fleet(spec: dict, args):
    """Assemble the (capped provider, launch, executor, configs) for a fleet run — the
    pure wiring, split out so it is unit-testable without renting hardware."""
    from run_farm import (
        CappedProvider,
        HostSpec,
        LaunchSpec,
        RentalLedger,
        RunPodProvider,
        VastProvider,
    )

    ledger = RentalLedger(args.ledger)
    if args.provider == "vast":
        # Vast injects sshd via runtype=ssh; the onstart only installs the engine.
        base = VastProvider(ledger=ledger)
        onstart = _vast_onstart()
    elif args.provider == "runpod":
        # RunPod runs the onstart as the pod's main process and injects no sshd, so the
        # onstart starts its own and authorizes the campaign key. SECURE (datacenter)
        # on-demand pods have reliable SSH — the flaky-proxy failure Vast community hosts
        # show does not occur — so this is the default tier.
        base = RunPodProvider(ledger=ledger, cloud_type=args.cloud_type.upper(),
                              interruptible=False)
        onstart = _runpod_onstart(_read_pubkey(args.ssh_key))
    else:
        raise ValueError(f"unknown provider: {args.provider}")
    provider = CappedProvider(base, args.cap_usd, ledger)   # refuses to overspend

    launch = LaunchSpec(image=args.image, onstart=onstart,
                        label=spec.get("gtag", "morpho-campaign"))
    host_spec = HostSpec(gpu_name=args.gpu, max_dph=args.max_dph, min_cuda=0.0)
    executor = FastFailExecutor(
        provider, RUN_FN_REF, launch, host_spec=host_spec,
        key_path=args.ssh_key, local_work_dir=args.out,
        config_class=CONFIG_CLASS_REF,
        # The default image installs the engine system-wide (no venv) and exposes it as
        # `python`; results sync under /root/runs. Both match the onstart builders.
        remote_python="python", remote_work_dir=_REMOTE_WORK_DIR,
        ssh_grace=args.ssh_grace, ready_timeout=args.ready_timeout,
        rent_timeout=args.rent_timeout, max_attempts=args.max_attempts)
    return provider, launch, executor, plan_configs(spec)


def cmd_fleet(spec: dict, args) -> int:  # pragma: no cover — rents real hardware
    """Run the campaign over a rented, teardown-verifying, dollar-capped fleet."""
    try:
        _provider, _launch, executor, configs = _build_fleet(spec, args)
    except (ValueError, FileNotFoundError) as e:
        print(str(e), file=sys.stderr)
        return 2

    print(f"launching {len(configs)} legs on {args.provider} "
          f"(cap ${args.cap_usd:.2f}); ledger -> {args.ledger}")
    try:
        # A ProviderExecutor is driven DIRECTLY (it registers + runs on the rented
        # box and syncs results back to local_work_dir) — NOT through run_campaign,
        # which is the in-process Executor-protocol path. Admission (require_gpu) is
        # enforced by the worker on the box, so none is passed here.
        results = executor.run(configs)
    finally:
        # Teardown is best-effort (§ run-farm README): always reap strays after.
        print("campaign done. Run `run-farm-reap` to catch any create-window "
              "orphans, and check the ledger:", args.ledger)
    for r in results:
        if r.get("error"):
            print(f"  {r.get('run')}: ERROR {r['error']}")
    _report_results(args.out, configs, subdir=_REMOTE_RUNS)
    return 0


# The synced-artifacts subdir. The worker writes DONE.json/checkpoints under
# `/root/{_REMOTE_RUNS}/<run>/`; `_sync_back` scp's that whole dir DOWN, so locally
# they land under `<out>/{_REMOTE_RUNS}/<run>/` — which `_report_results` must match.
_REMOTE_RUNS = "runs"
_REMOTE_WORK_DIR = f"/root/{_REMOTE_RUNS}"

# Install jax-morpho from the GitHub **tarball**, not `git+https`: pip fetches and builds
# it with no `git` binary, so the box needs no `apt-get install git` — and thus no
# `apt-get update`, whose slow Debian mirrors were the bootstrap's dominant cost (measured
# 2026-07-19: apt alone burned most of the 900s ready-timeout on marginal Vast hosts,
# forcing failover). The engine's deps (run-farm, jax, numpy, scipy) all resolve from
# PyPI, so the tarball is the only non-PyPI fetch. `main.tar.gz` tracks the default
# branch; pin `.../archive/refs/tags/vX.Y.Z.tar.gz` for a reproducible campaign. GPU-scale
# evolve campaigns override `--image` with a CUDA + Python-3.11 image and prepend
# `pip install 'jax[cuda12]'` (then `jax-morpho[scale] @ .../main.tar.gz`).
_INSTALL = ('pip install -q "jax-morpho @ '
            'https://github.com/JimGalasyn/jax-morpho/archive/refs/heads/main.tar.gz"')


def _vast_onstart() -> str:
    """Vast bootstrap. Vast's `runtype=ssh` injects its OWN sshd/proxy and keeps the
    container alive, so the onstart only needs to install the engine and echo the
    readiness marker `ProviderExecutor` probes for. Matched to `python:3.11-slim`
    (Python 3.11, the engine's floor; system-wide install, so `remote_python=python`)."""
    return f"#!/bin/bash\nset -e\n{_INSTALL}\necho ENGINE_READY\n"


def _runpod_onstart(pubkey: str) -> str:
    """RunPod bootstrap. Unlike Vast, RunPod runs THIS script as the pod's main process
    (`dockerStartCmd`) and maps container port 22 to a public port — it does NOT inject
    an sshd. So a bare image (`python:3.11-slim`, no sshd) is unreachable unless the
    onstart starts its own. We apt-install openssh (RunPod-datacenter mirrors are fast,
    so this is cheap — the Vast apt problem was community-host mirrors), authorize the
    campaign key (`--ssh-key`'s public half; RunPod does not inject account keys once the
    start command is overridden), start sshd, install the engine, and `sleep infinity` so
    the container — and thus sshd — stays up for the whole campaign (without it the script
    exits, PID 1 dies, and the pod is torn down mid-run)."""
    return (
        "#!/bin/bash\n"
        "set -e\n"
        "export DEBIAN_FRONTEND=noninteractive\n"
        "apt-get update -qq\n"
        "apt-get install -y -qq openssh-server >/dev/null\n"
        "mkdir -p /run/sshd /root/.ssh\n"
        f"echo {shlex.quote(pubkey)} > /root/.ssh/authorized_keys\n"
        "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys\n"
        "ssh-keygen -A >/dev/null 2>&1 || true\n"
        "sed -i 's/^#\\?PermitRootLogin.*/PermitRootLogin prohibit-password/' "
        "/etc/ssh/sshd_config\n"
        "/usr/sbin/sshd\n"
        f"{_INSTALL}\n"
        "echo ENGINE_READY\n"
        "sleep infinity\n"
    )


class FastFailExecutor(ProviderExecutor):
    """A `ProviderExecutor` that fails a dead host over in `ssh_grace` seconds instead of
    burning the full `ready_timeout` on it.

    The base `_wait_engine_ready` cannot tell "SSH refused, host is dead" from "SSH works,
    engine still installing" — both are a non-zero probe — so a host whose SSH never comes
    up (a flaky Vast proxy that never wires; ~half the marketplace in a bad window) costs
    the whole `ready_timeout` before failover. We split the wait: SSH must answer within
    `ssh_grace` (else fail over now), and only then does the engine-install poll get the
    full `ready_timeout`. Pair with a tight `rent_timeout` so a host stuck provisioning
    (never reaches "running") also fails fast."""

    def __init__(self, *args, ssh_grace: float = 150.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssh_grace = ssh_grace

    def _wait_engine_ready(self, host) -> None:
        # Phase 1: SSH reachability. Refused-and-refused-again => dead host, fail over.
        ssh_deadline = time.monotonic() + self.ssh_grace
        while time.monotonic() < ssh_deadline:
            rc, _ = _ssh(self.key_path, host.ssh_host, host.ssh_port, "echo OK",
                         timeout=15)
            if rc == 0:
                break
            time.sleep(8)
        else:
            raise HostProbeFailed(
                f"SSH never reachable on {host.id} within {self.ssh_grace:.0f}s "
                f"({host.ssh_host}:{host.ssh_port}) -> failover")
        # Phase 2: engine install (import the RunFn's module) within ready_timeout.
        check = f"{self.remote_python} -c 'import {self.engine_module}'"
        deadline = time.monotonic() + self.ready_timeout
        last = ""
        while time.monotonic() < deadline:
            rc, out = _ssh(self.key_path, host.ssh_host, host.ssh_port, check,
                           timeout=30)
            if rc == 0:
                return
            last = out
            time.sleep(10)
        raise HostProbeFailed(
            f"engine not ready on {host.id} within {self.ready_timeout:.0f}s: "
            f"{last[-200:]}")


def _report_results(out: str, configs, subdir: str = "") -> None:
    """Print each run's DONE.json summary — and, for the M-U arm, whether its
    known-answer gate passed (the validation that traveled with the campaign).

    `local` writes DONE.json to `<out>/<run>/`; the fleet path scp's the box's
    `/root/runs` dir down, so its artifacts land under `<out>/runs/<run>/` — pass
    `subdir="runs"` (i.e. `_REMOTE_RUNS`) so a *successful* fleet run isn't misreported
    as "(no result)". `subdir=""` (local) preserves the original layout."""
    base = Path(out) / subdir
    for c in configs:
        done = base / c.run_name() / "DONE.json"
        if not done.exists():
            print(f"  {c.run_name()}: (no result)")
            continue
        r = json.loads(done.read_text())     # DONE.json IS the RunFn's result dict
        if r.get("kind") == "mu_gate":
            print(f"  [M-U gate] worst angle(G)={r.get('worst_angle_G'):.1f}° "
                  f"median angle(P)={r.get('median_angle_P'):.1f}° "
                  f"-> {'PASS' if r.get('gate_pass') else 'FAIL'}")
        else:
            print(f"  {r.get('arm')}/r{r.get('replicate')}: "
                  f"het {r.get('heterozygosity_start')} -> "
                  f"{r.get('heterozygosity_final')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jax-morpho-campaign",
        description="Plan, cost, and launch an evodevo fleet campaign (run-farm).")
    p.add_argument("--spec", help="campaign spec JSON (default: built-in)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("plan", help="expand legs and show config hashes")

    pe = sub.add_parser("estimate", help="pre-launch GPU-h, wall-clock, and USD")
    pe.add_argument("--hosts", type=int, default=1, help="rented hosts (parallelism)")

    pl = sub.add_parser("local", help="run the campaign in-process (no cloud)")
    pl.add_argument("--out", default="campaign_out")

    pf = sub.add_parser("fleet", help="run over a rented, dollar-capped fleet")
    pf.add_argument("--provider", choices=["vast", "runpod"], required=True)
    pf.add_argument("--cap-usd", type=float, required=True,
                    help="hard dollar ceiling; renting past it is refused")
    pf.add_argument("--gpu", default="RTX_3090", help="HostSpec.gpu_name")
    pf.add_argument("--max-dph", type=float, default=0.40, help="$/hr ceiling per host")
    # RunPod only: SECURE = datacenter, reliable SSH (the default, so a first fleet run
    # just works); COMMUNITY is cheaper but has the flaky-SSH tail Vast community shows.
    pf.add_argument("--cloud-type", choices=["secure", "community"], default="secure",
                    help="RunPod tier (ignored for vast)")
    # The SSH key the executor connects with. For RunPod its PUBLIC half (<key>.pub) is
    # authorized on the pod by the onstart; for Vast the key must be registered with Vast.
    pf.add_argument("--ssh-key", default="~/.ssh/vastai",
                    help="private key path; <key>.pub is authorized on RunPod pods")
    # Fast-fail knobs: SSH must answer within --ssh-grace or the host is dead (fail over);
    # once up, the engine install gets --ready-timeout; --rent-timeout bounds provisioning
    # (a host stuck 'loading' past it fails over too). Defaults tuned on Vast 2026-07-19.
    pf.add_argument("--ssh-grace", type=float, default=150.0)
    pf.add_argument("--ready-timeout", type=float, default=480.0)
    pf.add_argument("--rent-timeout", type=float, default=240.0)
    pf.add_argument("--max-attempts", type=int, default=20,
                    help="offers to walk before giving up (cheap now that dead hosts "
                         "fail fast)")
    # Python 3.11 (the engine's floor) + pip, no CUDA hook (so no nvidia-container
    # create failures). Override with a CUDA + py3.11 image for GPU-scale campaigns.
    pf.add_argument("--image", default="python:3.11-slim")
    pf.add_argument("--ledger", default="campaign_out/ledger.jsonl")
    pf.add_argument("--out", default="campaign_out")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_spec(args.spec)
    return {
        "plan": cmd_plan, "estimate": cmd_estimate,
        "local": cmd_local, "fleet": cmd_fleet,
    }[args.cmd](spec, args)


if __name__ == "__main__":
    raise SystemExit(main())
