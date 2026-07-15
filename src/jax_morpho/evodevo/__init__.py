"""jax_morpho.evodevo — closing the genotype→development→selection loop.

Phase 0 is a faithful calibration against Milocco & Uller (2026 PNAS): their
toggle-switch developmental model (``reference_mu``) and the developmental-
sensitivity engine (``sensitivity``) are validated against their Fig 3C and
Fig 1C before any of our own developmental engine is substituted.
See docs/DESIGN.md.
"""
from __future__ import annotations

from jax_morpho.evodevo.reference_mu import (
    build_G_sensitivity,
    develop,
    develop_population,
    develop_theta,
    run_fig3c,
    simulate_fig3c,
    toggle_deriv,
)
from jax_morpho.evodevo.sensitivity import (
    forward_sensitivity,
    implicit_sensitivity,
    reverse_sensitivity,
)

__all__ = [
    # reference model (Milocco-Uller toggle switch)
    "develop", "develop_theta", "develop_population", "toggle_deriv",
    "simulate_fig3c", "run_fig3c", "build_G_sensitivity",
    # sensitivity engine
    "forward_sensitivity", "reverse_sensitivity", "implicit_sensitivity",
]
