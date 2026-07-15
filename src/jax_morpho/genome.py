"""Genome -> mechanical program -> tissue form (differentiable to the genome).

Amino-acid chemistry sets the mechanical program that shapes a tissue, and
because the whole relaxation is autodiff, gradients flow all the way back to
the genome — so a target FORM can be turned into the GENOME that produces it
(gradient-based evo-devo).

Developmental map (fixed genotype->phenotype rule):
  A "gene" is the hydrophobicity expressed along a body axis.  Hydrophobic
  expression tightens confinement on that axis (cells drawn together),
  shortening it; hydrophilic loosens it.  The contrast between the two
  axial genes sets the tissue's elongation, their sum sets its size:

      k_axis = k0 * exp(beta * h_axis),   V_ext = 1/2 (kx x^2 + ky y^2)

  The genome is (h_x, h_y).  Real amino acids instantiate it via their
  hydrophobicity class: hydrophobic -> +1, neutral -> 0, hydrophilic -> -1.

This deliberately uses a RESPONSIVE, smooth objective (shape via a
patterning field).  Sorting- or growth-driven morphologies need active or
stochastic dynamics before they are viable gradient targets (see
inverse_design notes).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jax_morpho.center_based import gyration_morphology
from jax_morpho.inverse_design import adam, relax_in_field

# Amino-acid hydrophobicity class (standard 20 residues + stop).
_AA_HYDROPHOBICITY: dict[str, str] = {
    "A": "hydrophobic", "C": "hydrophobic", "F": "hydrophobic",
    "I": "hydrophobic", "L": "hydrophobic", "M": "hydrophobic",
    "V": "hydrophobic", "W": "hydrophobic",
    "D": "hydrophilic", "E": "hydrophilic", "K": "hydrophilic",
    "N": "hydrophilic", "Q": "hydrophilic", "R": "hydrophilic",
    "G": "neutral", "H": "neutral", "P": "neutral",
    "S": "neutral", "T": "neutral", "Y": "neutral",
}
# hydrophobicity class -> axial expression scalar
_HYDRO = {"hydrophobic": 1.0, "neutral": 0.0, "hydrophilic": -1.0}

K0_DEFAULT = 0.12      # baseline confinement stiffness
BETA_DEFAULT = 0.9     # hydrophobicity -> stiffness gain


def hydro_scalar(aa: str) -> float:
    """Amino acid -> axial hydrophobicity expression in {-1, 0, +1}."""
    return _HYDRO.get(_AA_HYDROPHOBICITY.get(aa, "neutral"), 0.0)


def sequence_to_genome(aa_x: str, aa_y: str) -> jnp.ndarray:
    """Two amino acids (the axial genes) -> genome (h_x, h_y)."""
    return jnp.array([hydro_scalar(aa_x), hydro_scalar(aa_y)], dtype=jnp.float32)


def develop(genome, k0: float = K0_DEFAULT, beta: float = BETA_DEFAULT):
    """Genotype->phenotype: genome (h_x, h_y) -> confinement field
    parameters [log kx, log ky] for the center-based relaxation."""
    logk0 = jnp.log(k0)
    return jnp.stack([logk0 + beta * genome[0], logk0 + beta * genome[1]])


@jax.jit
def axial_sigmas(pos):
    """(sigma_x, sigma_y): per-axis spatial spread — orientation-sensitive,
    unlike the orientation-agnostic gyration aspect.  sigma_y > sigma_x
    means the tissue is elongated along y."""
    c = pos - pos.mean(0)
    return (jnp.sqrt((c[:, 0] ** 2).mean() + 1e-9),
            jnp.sqrt((c[:, 1] ** 2).mean() + 1e-9))


def genome_morphology(genome, pos0, relax_steps: int = 200,
                      k0: float = K0_DEFAULT, beta: float = BETA_DEFAULT):
    """Genome -> (size, aspect) of the developed tissue.  Differentiable."""
    params = develop(genome, k0, beta)
    return gyration_morphology(relax_in_field(pos0, params, relax_steps))


def fit_genome(pos0, size_target, aspect_target, g0=None, lr: float = 0.08,
               n_steps: int = 40, relax_steps: int = 200,
               k0: float = K0_DEFAULT, beta: float = BETA_DEFAULT):
    """Optimize the genome to produce a target (size, aspect) form.

    Gradients flow genome -> mechanical params -> relaxation -> form.
    Returns (genome, history).
    """
    if g0 is None:
        g0 = jnp.array([0.0, 0.0], dtype=jnp.float32)

    def loss(g):
        sz, ap = genome_morphology(g, pos0, relax_steps, k0, beta)
        return (((sz - size_target) / size_target) ** 2
                + ((ap - aspect_target) / aspect_target) ** 2)

    return adam(loss, g0, lr=lr, n_steps=n_steps)


def nearest_amino_acids(genome, palette=("F", "G", "K")) -> tuple[str, str]:
    """Map an optimized continuous genome back to amino acids: the palette
    member whose hydrophobicity class is closest per axis (default one
    hydrophobic, one neutral, one hydrophilic representative)."""
    scal = {aa: hydro_scalar(aa) for aa in palette}

    def closest(h):
        return min(scal, key=lambda aa: abs(scal[aa] - float(h)))

    return closest(genome[0]), closest(genome[1])
