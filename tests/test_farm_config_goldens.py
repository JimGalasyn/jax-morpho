"""Frozen leg->config translation: the byte-critical `reserved` tuple.

`morpho_leg_to_config` decides which leg keys become top-level RunConfig fields and
which ride in `params`. That split is expressed as `reserved=("seed",)` — and
`params` is serialized into `config_hash`, so getting the tuple wrong by one key
silently renames every run in the campaign (see test_run_config_goldens for why a
silent rename is the expensive failure). ``seed`` is consumed as a field; ``arm``
and ``replicate`` deliberately STAY in params so an aggregation can groupby them.

These literals pin the split. Do not regenerate them from the current code to make a
failing test pass — that pins the drift and proves nothing.
"""
from __future__ import annotations

import pytest

from jax_morpho.farm_config import morpho_leg_to_config

# Exercises every branch: explicit seed, unreserved passthrough knobs
# (selection/variation/sigma_env), the axis keys (arm/replicate), and
# campaign-authoritative keys the leg tries to spoof (gtag/required_shas).
LEG_EXPLICIT = {
    "rid": "point_r1",
    "arm": "point", "replicate": 1,
    "seed": 12345,
    "cfg": {"kind": "evolve", "n_pop": 120, "n_generations": 8, "n_genes": 4,
            "dtype": "float64"},
    "selection": {"type": "truncation", "frac": 0.2},
    "variation": {"type": "point", "rate": 0.02},
    "sigma_env": 0.005,
    "gtag": "SPOOFED", "required_shas": {"x": "SPOOFED"},
}


def test_explicit_leg_params_are_frozen():
    c = morpho_leg_to_config(LEG_EXPLICIT, "morpho-v1", {"engine": "abc123"})
    assert c.params == {
        "rid": "point_r1",
        "cfg": {"kind": "evolve", "n_pop": 120, "n_generations": 8,
                "n_genes": 4, "dtype": "float64"},
        "plan": [],
        "arm": "point", "replicate": 1,          # axes: ride in params (groupby)
        "selection": {"type": "truncation", "frac": 0.2},
        "variation": {"type": "point", "rate": 0.02},
        "sigma_env": 0.005,                       # unreserved knob -> passthrough
        "gtag": "morpho-v1",                      # authoritative, NOT "SPOOFED"
        "required_shas": {"engine": "abc123"},
    }
    # seed is consumed as a top-level field and must NOT also be in params
    assert "seed" not in c.params


def test_explicit_leg_fields_and_identity_are_frozen():
    c = morpho_leg_to_config(LEG_EXPLICIT, "morpho-v1", {"engine": "abc123"})
    assert (c.kind, c.n_pop, c.n_generations, c.n_genes, c.dtype, c.seed) == (
        "evolve", 120, 8, 4, "float64", 12345)
    assert c.config_hash() == "6b4ea1a80cca"
    assert c.run_name() == "evolve_N120_6b4ea1a80cca"


def test_campaign_authoritative_keys_are_unspoofable():
    """A leg carrying gtag/required_shas must not override the campaign's
    attestation identity — otherwise a leg could forge its own provenance."""
    c = morpho_leg_to_config(LEG_EXPLICIT, "morpho-v1", {"engine": "abc123"})
    assert c.params["gtag"] == "morpho-v1"
    assert c.params["required_shas"] == {"engine": "abc123"}


def test_config_does_not_alias_the_leg_cfg():
    """cfg is COPIED by leg_params: mutating the leg's cfg after the call must not
    change the config's identity. Legs commonly share one cfg object, so aliasing
    would let one leg silently rename another's run directory. (Passthrough knob
    dicts are NOT copied by leg_params — the CLI's build_legs deep-copies them
    before they reach here; morpho_leg_to_config's own guarantee is cfg/plan/
    required_shas, matching run_farm.farm.leg_params.)"""
    leg = {"rid": "r", "arm": "point", "replicate": 0, "seed": 1,
           "cfg": {"kind": "evolve", "n_pop": 60, "n_generations": 4, "n_genes": 4},
           "variation": {"type": "point", "rate": 0.02}}
    c = morpho_leg_to_config(leg, "g", {})
    before = c.config_hash()
    leg["cfg"]["n_pop"] = 999
    assert c.config_hash() == before


def test_required_shas_does_not_alias_the_campaign_baseline():
    """Per-leg COPY: mutating one config's required_shas must not reach into the
    campaign's verification baseline or any sibling leg."""
    shas = {"engine": "abc123"}
    c = morpho_leg_to_config(
        {"rid": "r", "seed": 0, "cfg": {"kind": "evolve", "n_pop": 60}}, "g", shas)
    c.params["required_shas"]["engine"] = "TAMPERED"
    assert shas == {"engine": "abc123"}
