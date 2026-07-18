"""jax_morpho.RunConfig must satisfy the run_farm.RunConfig Protocol.

The consumer relationship is STRUCTURAL and unenforced by imports: the campaign
layer (run-farm) types against a Protocol, and `jax_morpho.runs.RunConfig` happens
to satisfy it. Nothing makes that true at import time — a field rename or a `to_json`
"cleanup" in runs.py would compile, pass most tests, and then fail on a rented box
when the worker rebuilds the config from JSON. This test is the cheap thing standing
in that gap. If it fails, the run-farm seam is broken.
"""

from run_farm.protocols import RunConfig as RunConfigProtocol

from jax_morpho.runs import RunConfig


def test_run_config_satisfies_run_farm_protocol():
    c = RunConfig(kind="evolve", n_pop=16, n_generations=4, n_genes=4,
                  params={"arm": "point"})
    # isinstance works on a @runtime_checkable Protocol (presence-only).
    # NOTE: issubclass() would raise TypeError on a data-member Protocol — don't.
    assert isinstance(c, RunConfigProtocol)


def test_run_config_has_the_contract_surface():
    """The members run-farm actually reads/calls, spelled out so a rename here
    fails LOUDLY at CI rather than silently on a worker."""
    c = RunConfig(kind="mu_gate", dtype="float64", params={"k": 1})
    assert c.dtype == "float64"
    assert c.params == {"k": 1}
    assert isinstance(c.to_json(), str)
    assert RunConfig.from_json(c.to_json()) == c
    assert isinstance(c.config_hash(), str)
    assert c.run_name().endswith(c.config_hash())


def test_config_class_ref_rebuilds_remotely():
    """The remote path in miniature: a worker imports the config class by
    'module:ClassName' ref and rebuilds from JSON. Proves the ref the campaign CLI
    ships (`jax_morpho.runs:RunConfig`) resolves and round-trips to one identity."""
    from run_farm.remote import load_config_class
    cls = load_config_class("jax_morpho.runs:RunConfig")
    assert cls is RunConfig
    c = RunConfig(kind="evolve", n_pop=120, n_generations=8, n_genes=4,
                  params={"arm": "retro"})
    assert cls.from_json(c.to_json()).config_hash() == c.config_hash()


def test_dtype_default_is_float64():
    """evodevo needs x64 (implicit solve + Procrustes); run_farm.remote reads
    exactly this field to enable x64 on a fresh worker, so the default must be
    the width the engine actually runs in."""
    assert RunConfig().dtype == "float64"
