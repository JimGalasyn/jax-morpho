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
import sys
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
    estimate,
    run_campaign,
)

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


def cmd_fleet(spec: dict, args) -> int:  # pragma: no cover — rents real hardware
    """Run the campaign over a rented, teardown-verifying, dollar-capped fleet."""
    from run_farm import (
        CappedProvider,
        HostSpec,
        LaunchSpec,
        ProviderExecutor,
        RentalLedger,
        RunPodProvider,
        VastProvider,
    )

    ledger = RentalLedger(args.ledger)
    if args.provider == "vast":
        base = VastProvider(ledger=ledger)
    elif args.provider == "runpod":
        base = RunPodProvider(ledger=ledger, interruptible=True)
    else:
        print(f"unknown provider: {args.provider}", file=sys.stderr)
        return 2
    provider = CappedProvider(base, args.cap_usd, ledger)   # refuses to overspend

    launch = LaunchSpec(image=args.image, onstart=_ONSTART,
                        label=spec.get("gtag", "morpho-campaign"))
    host_spec = HostSpec(gpu_name=args.gpu, max_dph=args.max_dph)
    executor = ProviderExecutor(
        provider, RUN_FN_REF, launch, host_spec=host_spec,
        local_work_dir=args.out, config_class=CONFIG_CLASS_REF)

    registry = FileRunRegistry(args.out)
    sink = JsonlEventSink()
    admission = ProbeAdmission(require_gpu=True)
    configs = plan_configs(spec)
    print(f"launching {len(configs)} legs on {args.provider} "
          f"(cap ${args.cap_usd:.2f}); ledger -> {args.ledger}")
    try:
        run_campaign(configs, evodevo_run, registry=registry, sink=sink,
                     admission=admission, executor=executor)
    finally:
        # Teardown is best-effort (§ run-farm README): always reap strays after.
        print("campaign done. Run `run-farm-reap` to catch any create-window "
              "orphans, and check the ledger:", args.ledger)
    _report_results(args.out, configs)
    return 0


# Bootstrap a rented box: install run-farm + jax-morpho, then signal readiness the
# ProviderExecutor probes for. Kept minimal; override the image with --image.
_ONSTART = r"""#!/bin/bash
set -e
python -m venv /workspace/jaxenv
/workspace/jaxenv/bin/pip install -q --upgrade pip
/workspace/jaxenv/bin/pip install -q 'jax[cuda12]' \
  'jax-morpho[scale] @ git+https://github.com/JimGalasyn/jax-morpho.git'
echo "ENGINE_READY"
"""


def _report_results(out: str, configs) -> None:
    """Print each run's DONE.json summary — and, for the M-U arm, whether its
    known-answer gate passed (the validation that traveled with the campaign)."""
    base = Path(out)
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
    pf.add_argument("--image", default="nvidia/cuda:12.4.1-runtime-ubuntu22.04")
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
