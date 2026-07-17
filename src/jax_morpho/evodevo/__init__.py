"""jax_morpho.evodevo — closing the genotype→development→selection loop.

Phase 0 is a faithful calibration against Milocco & Uller (2026 PNAS): their
toggle-switch developmental model (``reference_mu``) and the developmental-
sensitivity engine (``sensitivity``) are validated against their Fig 3C and
Fig 1C before any of our own developmental engine is substituted.

Phase 1 replaces their ODE with the **mechanical** engine (``mechanical``): a
per-cell parameter field relaxed to a tissue equilibrium, differentiated by the
general fixed-point engine (``fixed_point``).

Phase 2 wires the rest of the spine: a nonlinear GRN genome map (``genome_map``),
a landmark + Procrustes shape readout (``phenotype``), the composed
genome→shape pipeline (``pipeline``), and the delta-method G (``quantgen``).

Phase 3 completes the quant-gen layer: a genetic architecture of diploid loci
(``genetics``), the one-generation response to selection (``response``), and the
two-solve ``Δz̄ = J M Jᵀβ`` path that needs no G and no J.

Phase 4 runs the loop over time (``evolution``): population → develop → select →
reproduce → repeat, with pluggable variation (point mutation, retroviral
insertion) and selection seams. Where Phase 3 asked "does `Δz̄ = Gβ` predict one
generation?", Phase 4 asks whether it *compounds* — and finds it does only while
genetic variance is healthy.

See docs/DESIGN.md.
"""
from __future__ import annotations

from jax_morpho.evodevo.fixed_point import (
    energy_sensitivity,
    finite_difference_sensitivity,
    fixed_point_sensitivity,
    implicit_jvp,
    implicit_vjp,
    project_out,
    rigid_modes,
)
from jax_morpho.evodevo.genetics import (
    Architecture,
    allele_frequencies,
    genome_from_scores,
    make_architecture,
    recombine,
    sample_environment,
)
from jax_morpho.evodevo.genetics import sample_genotypes as sample_genotypes_hwe
from jax_morpho.evodevo.response import (
    reference_optimum,
    run_sweep,
    simulate_response,
)
from jax_morpho.evodevo.evolution import (
    VariationContext,
    compose,
    effective_population_size,
    evolve,
    heterozygosity,
    mendelian_mating,
    neutral_selection,
    no_variation,
    point_mutation,
    retroviral_insertion,
    truncation_toward,
)
from jax_morpho.evodevo.genome_map import GRN, grn_field, grn_jacobian, init_grn
from jax_morpho.evodevo.mechanical import (
    equilibrate,
    field_morse_energy,
    force_residual,
    hex_blob,
    CONTACT_CUTOFF,
    contact_topology,
    pack_theta,
    unpack_theta,
    uniform_theta,
)
from jax_morpho.evodevo.mechanical import develop as develop_mechanical
from jax_morpho.evodevo.phenotype import (
    centroid_size,
    landmarks,
    make_reference,
    procrustes_align,
    procrustes_shape,
    shape_dim,
    shape_jacobian,
    tangent_basis,
    tangent_coords,
)
from jax_morpho.evodevo.pipeline import (
    Organism,
    lande_response_vjp,
    make_organism,
    phenotype_jacobian,
    phenotype_jvp,
    phenotype_population,
    phenotype_vjp,
    theta_of,
)
# Renamed on the way out, to avoid three separate collisions on this package's
# namespace (all pinned by tests/test_evodevo_api.py):
#   * bare ``phenotype``/``develop`` would shadow the ``phenotype`` submodule and
#     ``mechanical.develop``;
#   * ``develop_population`` already belongs to ``reference_mu`` — the published
#     v0.2.0 API — and being imported later it would silently *win*, handing
#     callers the toggle-switch developer where they asked for the mechanical one.
from jax_morpho.evodevo.pipeline import develop as develop_genome
from jax_morpho.evodevo.pipeline import develop_population as develop_genome_population
from jax_morpho.evodevo.pipeline import phenotype as phenotype_of
from jax_morpho.evodevo.quantgen import (
    angle_deg,
    average_effects,
    build_G,
    build_G_alleles,
    empirical_covariance,
    lande_response,
    relative_difference,
    selection_gradient,
    truncation_select,
)
from jax_morpho.evodevo.reference_mu import (
    build_G_sensitivity,
    develop_theta,
    run_fig3c,
    simulate_fig3c,
    toggle_deriv,
)
# BREAKING vs v0.2.0, deliberately: the bare names ``develop`` and
# ``develop_population`` are gone. They resolved to the Milocco-Uller *toggle
# switch ODE* — correct when evodevo was only the Phase-0 calibration, and a trap
# now that the package is headlined by the mechanical engine. Someone reaching
# for ``evodevo.develop_population`` today means "develop a population of
# organisms" and would have silently received the ODE toy.
#
# They now carry a ``_mu`` suffix and the mechanical/pipeline versions carry
# their own. Nothing is bound to the bare names, so v0.2.0 code fails with an
# immediate AttributeError naming the missing symbol — a loud break, which is the
# whole point. Preserving the name would have kept a silent wrong answer for API
# tidiness. Pinned by tests/test_evodevo_api.py.
from jax_morpho.evodevo.reference_mu import develop as develop_mu
from jax_morpho.evodevo.reference_mu import develop_population as develop_population_mu
from jax_morpho.evodevo.sensitivity import (
    forward_sensitivity,
    implicit_sensitivity,
    reverse_sensitivity,
)

__all__ = [
    # reference model (Milocco-Uller toggle switch) — *_mu, never bare: see the
    # import block above on why `develop` / `develop_population` were removed.
    "develop_mu", "develop_theta", "develop_population_mu", "toggle_deriv",
    "simulate_fig3c", "run_fig3c", "build_G_sensitivity",
    # sensitivity engine
    "forward_sensitivity", "reverse_sensitivity", "implicit_sensitivity",
    # fixed-point sensitivity engine (Phase 1)
    "fixed_point_sensitivity", "energy_sensitivity",
    "finite_difference_sensitivity", "rigid_modes", "project_out",
    # mechanical development (Phase 1)
    "field_morse_energy", "force_residual", "equilibrate", "develop_mechanical",
    "pack_theta", "unpack_theta", "uniform_theta", "hex_blob",
    "contact_topology", "CONTACT_CUTOFF",
    # genome map — layer A (Phase 2)
    "GRN", "init_grn", "grn_field", "grn_jacobian",
    # phenotype — layer C (Phase 2)
    "landmarks", "procrustes_align", "procrustes_shape", "make_reference",
    "shape_jacobian", "shape_dim", "centroid_size",
    "tangent_basis", "tangent_coords",
    # composed pipeline (Phase 2)
    "Organism", "make_organism", "theta_of", "develop_genome", "phenotype_of",
    "phenotype_jacobian", "develop_genome_population", "phenotype_population",
    # the two-solve reverse-mode path (Phase 3)
    "phenotype_vjp", "phenotype_jvp", "lande_response_vjp",
    "implicit_vjp", "implicit_jvp",
    # genetic architecture (Phase 3)
    "Architecture", "make_architecture", "sample_genotypes_hwe",
    "genome_from_scores", "sample_environment", "allele_frequencies", "recombine",
    # quantitative genetics — layer D (Phases 2-3)
    "build_G", "empirical_covariance", "relative_difference",
    "average_effects", "build_G_alleles", "selection_gradient",
    "lande_response", "truncation_select", "angle_deg",
    # the one-generation response protocol (Phase 3)
    "simulate_response", "run_sweep", "reference_optimum",
    # the evolution loop — layer E (Phase 4)
    "evolve", "heterozygosity", "effective_population_size",
    "mendelian_mating", "truncation_toward", "neutral_selection",
    "VariationContext", "no_variation", "point_mutation",
    "retroviral_insertion", "compose",
]
