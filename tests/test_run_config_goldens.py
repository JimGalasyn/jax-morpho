"""Frozen run identities: `RunConfig.to_json()` bytes are permanent directory names.

`config_hash` names a run's directory, and the registry's idempotent skip
(mechanism A) resolves a prior run by that name. So a change to the config's
serialization silently renames every run: `is_complete` stops recognizing finished
work, a resumed campaign restarts from zero, and a rented fleet re-bills for results
already on disk. Nothing raises; it just quietly costs money and time.

These literals pin the identity from the day the campaign layer was wired in.
**They do not get updated to match new behavior; they get obeyed.** A failure means
run identity drifted — fix the code, not the fixture.
"""
from __future__ import annotations

import json

from jax_morpho.runs import RunConfig, run_dir

# One fully-specified config, captured when the seam was built. The bytes come
# first in the assertions because they are the root cause of the hash and name.
GOLDEN_CONFIG = dict(kind="evolve", n_pop=120, n_generations=8, n_genes=4,
                     dtype="float64", seed=7,
                     params={"arm": "point", "replicate": 1,
                             "selection": {"type": "truncation", "frac": 0.2},
                             "variation": {"type": "point", "rate": 0.02}})
GOLDEN_JSON = ('{"dtype": "float64", "kind": "evolve", "n_generations": 8, '
               '"n_genes": 4, "n_pop": 120, "params": {"arm": "point", '
               '"replicate": 1, "selection": {"frac": 0.2, "type": "truncation"}, '
               '"variation": {"rate": 0.02, "type": "point"}}, "seed": 7}')
GOLDEN_HASH = "ef630e15da56"
GOLDEN_NAME = "evolve_N120_ef630e15da56"


def test_run_identity_is_frozen():
    c = RunConfig(**GOLDEN_CONFIG)
    assert c.to_json() == GOLDEN_JSON        # bytes first: fails most usefully
    assert c.config_hash() == GOLDEN_HASH
    assert c.run_name() == GOLDEN_NAME


def test_from_json_round_trips_to_the_same_identity():
    """The remote path: the driver serializes, the box deserializes, and both must
    agree on the run directory or the box writes results the driver never reads."""
    c = RunConfig.from_json(GOLDEN_JSON)
    assert c.config_hash() == GOLDEN_HASH
    assert c.run_name() == GOLDEN_NAME
    assert c.to_json() == GOLDEN_JSON        # round-trip must be byte-stable


def test_config_hash_is_insensitive_to_params_key_order():
    """Two configs differing only in dict insertion order are the SAME run — a
    worker rebuilding params from JSON gets the serializer's order and must not
    fork the identity."""
    a = RunConfig(kind="evolve", n_pop=8, params={"z": 1, "a": 2})
    b = RunConfig(kind="evolve", n_pop=8, params={"a": 2, "z": 1})
    assert a.config_hash() == b.config_hash()


def test_run_dir_writes_a_manifest_row(tmp_path):
    """run_dir's registry side effect (the MANIFEST line) is what a later reader
    parses to find prior work; its shape must stay stable."""
    c = RunConfig(**GOLDEN_CONFIG)
    d = run_dir(tmp_path, c)
    assert d.name == GOLDEN_NAME
    rows = [json.loads(ln) for ln in
            (tmp_path / "MANIFEST.jsonl").read_text().splitlines() if ln.strip()]
    assert rows == [{"run": GOLDEN_NAME, "config": json.loads(GOLDEN_JSON)}]
