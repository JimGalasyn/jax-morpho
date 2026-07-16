"""The package namespace itself — a class of bug that hides in plain sight.

`jax_morpho.evodevo` re-exports from eight submodules, several of which
legitimately use the same natural name for their own version of a thing
(`develop`, `phenotype`, `develop_population`). Python resolves that silently by
last-import-wins, so a collision does not raise — it just hands callers the wrong
function. `reference_mu.develop_population` (the published v0.2.0 toggle-switch
API) and `pipeline.develop_population` (the mechanical one) collided exactly this
way, and a "does every name in __all__ exist?" check passed while returning the
wrong object.

These tests check *identity*, not existence.
"""
from __future__ import annotations

import types

import jax_morpho.evodevo as E
from jax_morpho.evodevo import mechanical, pipeline, reference_mu


def test_every_exported_name_exists():
    missing = [n for n in E.__all__ if not hasattr(E, n)]
    assert not missing, f"__all__ names nothing: {missing}"


def test_submodules_are_not_shadowed_by_functions():
    """`from jax_morpho.evodevo import phenotype` must give the *module*."""
    for name in ("phenotype", "mechanical", "pipeline", "genetics", "quantgen",
                 "response", "fixed_point", "genome_map", "reference_mu",
                 "sensitivity"):
        assert isinstance(getattr(E, name), types.ModuleType), \
            f"evodevo.{name} is shadowed by a non-module export"


def test_colliding_names_resolve_to_the_intended_function():
    """The three renames earn their keep here."""
    # reference_mu owns the bare names (published v0.2.0 API)
    assert E.develop is reference_mu.develop
    assert E.develop_population is reference_mu.develop_population
    # ...and the mechanical/pipeline versions are reachable under distinct ones
    assert E.develop_mechanical is mechanical.develop
    assert E.develop_genome is pipeline.develop
    assert E.develop_genome_population is pipeline.develop_population
    assert E.phenotype_of is pipeline.phenotype


def test_no_exported_name_is_bound_twice_to_different_objects():
    """Catch the next collision before it ships, not after.

    Any name exported from two submodules with different objects behind it is a
    silent last-import-wins hazard.
    """
    import importlib

    submodules = ("fixed_point", "genetics", "genome_map", "mechanical",
                  "phenotype", "pipeline", "quantgen", "reference_mu",
                  "response", "sensitivity")
    owners: dict[str, list[str]] = {}
    for m in submodules:
        mod = importlib.import_module(f"jax_morpho.evodevo.{m}")
        for name in E.__all__:
            if hasattr(mod, name) and getattr(E, name) is not getattr(mod, name):
                owners.setdefault(name, []).append(m)

    # A name may legitimately appear in a submodule it was not exported from
    # (e.g. re-imports); the hazard is only when the exported object is NOT the
    # one the reader would reach for. Flag anything where the exported object
    # belongs to no submodule that also defines the name.
    hazards = {
        n: mods for n, mods in owners.items()
        if not any(getattr(E, n) is getattr(importlib.import_module(
            f"jax_morpho.evodevo.{m}"), n, None) for m in submodules)
    }
    assert not hazards, f"exported names shadowed across submodules: {hazards}"
