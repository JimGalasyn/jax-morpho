"""jax_morpho.evodevo — closing the genotype→development→selection loop.

Phase 0 is a faithful calibration against Milocco & Uller (2026 PNAS): their
toggle-switch developmental model and Figure 3C breeder's-equation test are
reproduced in ``reference_mu`` before any of our own developmental engine is
substituted. See docs/DESIGN.md.
"""
from __future__ import annotations

from jax_morpho.evodevo.reference_mu import (
    develop,
    develop_population,
    run_fig3c,
    simulate_fig3c,
)

__all__ = ["develop", "develop_population", "simulate_fig3c", "run_fig3c"]
