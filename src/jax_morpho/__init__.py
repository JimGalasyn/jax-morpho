"""jax-morpho: differentiable, GPU-scale developmental morphogenesis in JAX.

A center-based tissue engine where cells are points with a Morse
adhesion/repulsion potential, relaxation is autodiff gradient descent, and
growth is cell division. It reproduces real-epithelium topology statistics,
scales to millions of cells on GPU via neighbor lists, and maps an evolvable
genome to tissue form — with gradients that reach the genome, enabling
gradient-based inverse design of morphology.

The neighbor-list scaling layer (``jax_morpho.scale``) needs the optional
``jax_md`` dependency and is imported on demand, not here.
"""
from __future__ import annotations

__version__ = "0.1.1"

from jax_morpho.center_based import (
    A_DEFAULT,
    D_DEFAULT,
    R_EQ_DEFAULT,
    R_MAX_DEFAULT,
    divide_cells,
    grow_relax,
    gyration_morphology,
    interior_side_counts,
    morse_energy,
    relax,
)
from jax_morpho.inverse_design import (
    adam,
    confinement_energy,
    fit_morphology,
    morphology_loss,
    relax_in_field,
)
from jax_morpho.genome import (
    develop,
    fit_genome,
    genome_morphology,
    hydro_scalar,
    nearest_amino_acids,
    sequence_to_genome,
)
from jax_morpho.stats import (
    GIBSON_EPITHELIUM_SIDES,
    POISSON_VORONOI_SIDES,
    l1_distance,
    mean_sides,
    side_distribution,
)

__all__ = [
    "__version__",
    # center-based engine
    "morse_energy", "relax", "divide_cells", "grow_relax",
    "interior_side_counts", "gyration_morphology",
    "D_DEFAULT", "A_DEFAULT", "R_EQ_DEFAULT", "R_MAX_DEFAULT",
    # inverse design
    "confinement_energy", "relax_in_field", "morphology_loss",
    "fit_morphology", "adam",
    # genome -> form
    "sequence_to_genome", "develop", "genome_morphology", "fit_genome",
    "hydro_scalar", "nearest_amino_acids",
    # topology statistics
    "POISSON_VORONOI_SIDES", "GIBSON_EPITHELIUM_SIDES", "l1_distance",
    "side_distribution", "mean_sides",
]
