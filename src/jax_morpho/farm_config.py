"""The `FarmCampaign` config factory for this engine's config shape.

When the campaign layer was extracted to run-farm, the physics-agnostic half of
``leg -> config`` became `run_farm.farm.leg_params` (copy semantics + the
unspoofable campaign-authoritative keys); the morpho-shaped field assignment is
here as the engine's `config_factory`.

Pass `morpho_leg_to_config` to `FarmCampaign(config_factory=...)` (or hand it to the
campaign CLI) to plan a farm over `jax_morpho.RunConfig` instead of run-farm's
default `SimpleRunConfig`.
"""

from __future__ import annotations

from run_farm.farm import leg_params

from jax_morpho.runs import RunConfig


def morpho_leg_to_config(leg: dict, gtag: str, required_shas: dict) -> RunConfig:
    """A farm leg -> a `jax_morpho.RunConfig` (the FarmCampaign config_factory).

    ⚠ ``reserved=("seed",)`` is BYTE-CRITICAL. The factory reads the leg's top-level
    ``seed`` as a dataclass field, so ``seed`` must NOT also ride in ``params`` — but
    ``arm`` and ``replicate`` (also top-level leg keys from `run_farm.sweep.legs`)
    deliberately DO stay in ``params`` so an aggregation can ``groupby`` them off the
    result records. Any drift in this set changes ``params``, which changes
    ``config_hash``, which renames every run in the campaign. Pinned byte-for-byte by
    tests/test_farm_config_goldens.py.

    The engine fields (``kind``, ``n_pop``, ``n_generations``, ``n_genes``,
    ``dtype``) come from the leg's ``cfg`` (the sweep grid cell); everything else in
    ``cfg`` — the developmental knobs, the selection/variation seams — rides through
    ``params['cfg']`` to the RunFn unchanged.
    """
    params = leg_params(leg, gtag, required_shas, reserved=("seed",))
    cfg = params["cfg"]                        # the same copy leg_params already made
    return RunConfig(
        kind=str(cfg.get("kind", "evolve")),
        n_pop=int(cfg.get("n_pop", 0)),
        n_generations=int(cfg.get("n_generations", 0)),
        n_genes=int(cfg.get("n_genes", 0)),
        dtype=str(cfg.get("dtype", "float64")),
        seed=int(leg.get("seed", 0)),
        params=params)
