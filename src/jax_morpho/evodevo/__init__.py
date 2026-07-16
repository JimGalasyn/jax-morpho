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
]
