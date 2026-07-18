"""Restartable, registered runs — the campaign identity of one developmental run.

`RunConfig` is the single source of truth for a run: serialized into every output
and hashed into the run directory name. It **structurally satisfies the
`run_farm.RunConfig` Protocol** (pinned by tests/test_farm_conformance.py), so the
extracted campaign layer drives jax-morpho lineages without importing anything
morpho-specific. Its `to_json` bytes are the permanent names of every run directory
in every campaign ledger ever written — this class does not change; its identity is
frozen (tests/test_run_config_goldens.py).

The full-state checkpoint helpers (`save_checkpoint`/`load_checkpoint`/`run_dir`)
were physics-agnostic and moved WHOLESALE to `run_farm.config` when the campaign
layer was extracted from jax-solitons (2026-07). They are re-exported here so call
sites keep working; `load_checkpoint` is bound to this engine's `RunConfig` so it
rebuilds the concrete type rather than run-farm's default `SimpleRunConfig`.

A morpho run is one of two `kind`s (see runfns.evodevo_run):

  * ``"evolve"`` — one lineage: a population developed, selected, and reproduced
    for ``n_generations`` (the Gould replay-the-tape unit run_farm.sweep is built
    for). Full state = the genotype ``scores`` array; a preempted lineage resumes
    at the generation it checkpointed, with a per-generation-derived RNG so the
    resumed stream is bit-identical to the uninterrupted one.
  * ``"mu_gate"`` — the Milocco–Uller Fig 3C known-answer gate (reference_mu),
    the built-in calibration arm DESIGN.md §5b asks every campaign to carry.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

# Moved to run-farm at extraction; re-exported so `jax_morpho.runs.<helper>` still
# resolves. Orbax replaces this checkpoint layer when sharded multi-device arrays
# land — that roadmap is now run-farm's.
from run_farm.config import run_dir, save_checkpoint
from run_farm.config import load_checkpoint as _load_checkpoint


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Declarative description of one morpho run.

    `params` carries the arm/replicate axes and the developmental knobs; top-level
    fields are the invariants every run has. Replaces per-script argparse.

    ``dtype`` defaults to ``"float64"``: the implicit fixed-point solve and the
    Procrustes readout need x64 to stay at the measurement noise floor, and
    ``run_farm.remote.run_one`` reads exactly this field to enable x64 on a fresh
    worker before any array exists (an in-process caller sets it itself).
    """

    kind: str = "evolve"          # "evolve" | "mu_gate"
    n_pop: int = 0                # population size (evolve); 0 for mu_gate
    n_generations: int = 0        # lineage length (evolve); 0 for mu_gate
    n_genes: int = 0              # GRN input genes (evolve)
    dtype: str = "float64"
    seed: int = 0
    params: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> str:
        # sort_keys is load-bearing, not cosmetic: a worker rebuilding params from
        # JSON gets whatever order the serializer emitted, and an order-sensitive
        # hash would fork one run's identity between driver and box.
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "RunConfig":
        return cls(**json.loads(s))

    def config_hash(self, n: int = 12) -> str:
        """Stable short hash for run-directory naming (mechanism A)."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:n]

    def run_name(self) -> str:
        return f"{self.kind}_N{self.n_pop}_{self.config_hash()}"


def load_checkpoint(path) -> tuple[dict, RunConfig, int]:
    """Read a checkpoint back: (state dict, RunConfig, step).

    `run_farm.config.load_checkpoint` bound to this engine's config type, so a
    restored checkpoint comes back as a `jax_morpho.RunConfig`, not run-farm's
    default `SimpleRunConfig`.
    """
    return _load_checkpoint(path, config_class=RunConfig)


# `save_checkpoint` and `run_dir` are re-exported unchanged from run_farm.config
# (imported at module top); they never read a morpho-specific field.
__all__ = ["RunConfig", "save_checkpoint", "load_checkpoint", "run_dir"]
