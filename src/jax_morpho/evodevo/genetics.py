"""The genetic architecture: diploid loci → the genome vector the GRN reads.

Phase 2's genome is a continuous vector `a` — fine for a Jacobian, useless for
population genetics, which is about *alleles at loci and their frequencies*.
Milocco & Uller's Fig 3C is a sweep over **minor allele frequency**, so
reproducing its pattern with our development needs the same substrate they have:
loci with additive allelic effects, Hardy–Weinberg sampling, and Mendelian
recombination.

    genotype scores  g ∈ {−1, 0, 1}   (n_loci, n_ind)     diploid, additive
    allelic effects  γ                (n_loci,)
    gene value       a_j = Σ_{l ∈ loci(j)} g_l γ_l

This mirrors `reference_mu`'s `th = (gs * gamma[:, None]).sum(0)` exactly, but
generalised to *many* genes rather than their two parameters. `reference_mu`
keeps its own byte-identical copies of the sampling and recombination helpers —
its numbers are the Phase 0 calibration and must not move.

Environment vs genome
---------------------
The developmental map's input is `[genome, environment]`; only the genome half is
heritable. That split is Milocco & Uller's (their `f` takes θ genetic and `u`
environmental) and it is not cosmetic: **without environmental variance,
P = G exactly and Fig 3C has nothing to show.** The whole result is that P
misaligns with the response where G does not — which requires the two to differ.
See :func:`sample_environment`.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Architecture(NamedTuple):
    """Which loci write to which gene, and with what effect.

    ``gene_of_locus[l]`` is the gene index locus ``l`` contributes to;
    ``gamma[l]`` is its additive allelic effect. Loci are unlinked.
    """
    gene_of_locus: np.ndarray      # (n_loci,) int
    gamma: np.ndarray              # (n_loci,) float
    n_genes: int

    @property
    def n_loci(self):
        return self.gene_of_locus.shape[0]


def make_architecture(n_genes, loci_per_gene, sigma_gamma, rng):
    """Random additive architecture: ``loci_per_gene`` unlinked loci per gene,
    allelic effects ~ N(0, sigma_gamma)."""
    gene_of_locus = np.repeat(np.arange(n_genes), loci_per_gene)
    gamma = rng.normal(0.0, sigma_gamma, gene_of_locus.shape[0])
    return Architecture(gene_of_locus=gene_of_locus, gamma=gamma, n_genes=n_genes)


def sample_genotypes(p, n_loci, n_ind, rng):
    """Genotype scores {−1, 0, 1} at Hardy–Weinberg frequencies [q², 2pq, p²].

    ``p`` is the minor-allele frequency — the axis Fig 3C sweeps.
    """
    q = 1.0 - p
    return rng.choice([-1, 0, 1], size=(n_loci, n_ind), p=[q * q, 2 * p * q, p * p])


def genome_from_scores(scores, arch: Architecture):
    """Genotype scores ``(n_loci, n_ind)`` → genome vectors ``(n_ind, n_genes)``.

    Additive within a gene: ``a_j = Σ_{l ∈ loci(j)} g_l γ_l``.
    """
    contrib = scores * arch.gamma[:, None]                  # (n_loci, n_ind)
    a = np.zeros((arch.n_genes, scores.shape[1]))
    np.add.at(a, arch.gene_of_locus, contrib)
    return a.T                                              # (n_ind, n_genes)


def sample_environment(n_ind, n_env, sigma_env, rng):
    """Non-heritable developmental inputs ``(n_ind, n_env)``.

    Load-bearing, not decoration: with ``sigma_env = 0`` the phenotypic
    covariance P *equals* G and the Fig-3C contrast (G predicts, P misaligns)
    cannot exist. Milocco & Uller's `u` plays the same role.
    """
    return rng.normal(0.0, sigma_env, (n_ind, n_env))


def allele_frequencies(scores):
    """Realised (p, q) per locus from genotype scores — counts, not the nominal
    sampling frequency, so drift in a finite sample is accounted for."""
    n_A2 = (scores == 1).sum(1) * 2 + (scores == 0).sum(1)
    n_A1 = (scores == -1).sum(1) * 2 + (scores == 0).sum(1)
    tot = n_A1 + n_A2
    return n_A2 / tot, n_A1 / tot


def recombine(parentA, parentB, n_off, rng):
    """Vectorised Mendelian recombination, unlinked loci.

    A ``{−1, 0, 1}`` score is a plus-allele count minus 1, so a parent transmits
    a gamete ~ Bernoulli(count/2) and the offspring score is
    ``gameteA + gameteB − 1``. ``parentA``/``parentB``: ``(n_loci, n_pairs)``.
    Returns ``(n_loci, n_pairs * n_off)``.
    """
    ca = np.repeat(parentA, n_off, axis=1) + 1
    cb = np.repeat(parentB, n_off, axis=1) + 1
    return rng.binomial(1, ca / 2.0) + rng.binomial(1, cb / 2.0) - 1
