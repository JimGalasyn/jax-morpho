"""Layer E: the evolution loop — develop, select, reproduce, repeat.

This is the layer DESIGN.md §2E calls "the science artifact and the game loop":
the point where Phases 1–3 stop being a measurement of one generation and become
a population moving through time.

    scores → genome → develop → shape → select → mate → vary → scores'

Phase 3 asked *does `Δz̄ = Gβ` predict one generation?* (yes — G's error sits at
the measurement noise floor). Phase 4 asks the question that only exists over
time: **does that prediction compound?** It need not. `G` is a function of the
allele frequencies, and selection changes them — so a `G` measured at generation
0 describes a population that no longer exists by generation 10. Gate #4 is
whether the iterated breeder's equation tracks the realised trajectory, and
whether the frozen one visibly does not.

The seams (DESIGN.md §2E, §5c)
------------------------------
Two are kept deliberately separate, because they answer different questions:

* **mating** — the *reproduction model*: who pairs with whom and how alleles are
  transmitted. Default is Mendelian recombination with unlinked loci.
* **variation** — the *variation operator*: what happens to an offspring genome
  beyond inheritance. Point mutation is one. **Retroviral insertion is another**
  (:func:`retroviral_insertion`), and §5c requires this stay a seam rather than a
  hardcoded step, because the punctuation experiment is exactly a swap of this
  argument. It receives a :class:`VariationContext` carrying the architecture and
  a **donor pool**, since horizontal transfer needs somewhere to transfer *from*.

Effective population size is ~2N, not N
---------------------------------------
Worth knowing before reading any drift number out of this loop. The default
mating pairs the selected parents and gives **every pair exactly two offspring**,
so every parent contributes exactly two gametes and family-size variance is zero.
That is the classic equal-family-size case, `Ne ≈ 2N − 1` — measured here at
`Ne/N` = 2.02, 2.02, 2.00 for N = 20, 40, 80. Drift is therefore **half as strong**
as naive Wright-Fisher would suggest. This is a property of the reproduction
model, not of the genetics; a mating function with Poisson family sizes would
recover `Ne ≈ N`. Pinned by ``tests/test_evolution.py``.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import numpy as np

from jax_morpho.evodevo import genetics as GEN
from jax_morpho.evodevo import phenotype as PH
from jax_morpho.evodevo import quantgen as QG
from jax_morpho.evodevo import response as RS


# ---------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------

class VariationContext(NamedTuple):
    """What a variation operator is allowed to see.

    ``donors`` is the seam §5c requires: horizontal transfer is meaningless
    inside a single panmictic pool, so an operator that models it needs a genome
    pool from *elsewhere* to transfer from. It is ``None`` for a closed
    population, which is what makes the requirement visible rather than implied.
    """
    arch: GEN.Architecture
    parents: np.ndarray                 # (n_loci, n_selected) who reproduced
    donors: Optional[np.ndarray] = None  # (n_loci, n_donors) another lineage


#: A variation operator: ``(offspring_scores, ctx, rng) -> offspring_scores``.
VariationFn = Callable[[np.ndarray, VariationContext, np.random.Generator],
                       np.ndarray]


def no_variation(offspring, ctx, rng):
    """Inheritance only — the null operator."""
    return offspring


def point_mutation(rate):
    """Gradualism: each locus independently re-drawn with probability ``rate``.

    The *small* perturbation of §5c's contrast — it moves a genome a short
    distance and (per Phase 2/3) stays inside a developmental basin, where G
    predicts the response.
    """
    def op(offspring, ctx, rng):
        hit = rng.random(offspring.shape) < rate
        draw = rng.choice([-1, 0, 1], size=offspring.shape, p=[0.25, 0.5, 0.25])
        return np.where(hit, draw, offspring)
    return op


def retroviral_insertion(rate, genes_per_event=1):
    """Punctuation: overwrite a whole gene's loci **from a donor lineage**.

    The other half of §5c's contrast, and deliberately *not* a local
    perturbation: an event replaces every locus of one or more genes wholesale
    with a donor's alleles — horizontal transfer, not mutation. That is a large,
    coordinated, multi-locus jump, which is the regime Phase 2 measured as
    **basin-crossing**: the phenotype is discontinuous in the genome there and no
    local Jacobian, G included, describes the transition.

    ``Architecture.gene_of_locus`` supplies the block structure the event needs,
    which is why §5c asked for it to be preserved.

    This is the *operator*, not the experiment. It exists so the seam is real and
    exercised rather than asserted; the punctuation study (does viral
    macromutation reach basins gradualism cannot?) is downstream work.
    """
    def op(offspring, ctx, rng):
        if not 1 <= genes_per_event <= ctx.arch.n_genes:
            raise ValueError(
                f"genes_per_event={genes_per_event} out of range for an "
                f"architecture with {ctx.arch.n_genes} genes (need 1..n_genes)")
        if ctx.donors is None or ctx.donors.shape[1] == 0:
            raise ValueError(
                "retroviral_insertion needs a donor pool — horizontal transfer "
                "has to transfer from somewhere (VariationContext.donors is "
                "None). Give the loop a donor lineage, or use point_mutation.")
        out = offspring.copy()
        n_off = out.shape[1]
        infected = np.where(rng.random(n_off) < rate)[0]
        for i in infected:
            genes = rng.choice(ctx.arch.n_genes, size=genes_per_event,
                               replace=False)
            donor = ctx.donors[:, rng.integers(ctx.donors.shape[1])]
            block = np.isin(ctx.arch.gene_of_locus, genes)
            out[block, i] = donor[block]          # the whole gene, wholesale
        return out
    return op


def compose(*ops):
    """Apply variation operators in order — e.g. mutation *and* insertion."""
    def op(offspring, ctx, rng):
        for f in ops:
            offspring = f(offspring, ctx, rng)
        return offspring
    return op


# ---------------------------------------------------------------------------
# Mating (the reproduction model — a separate seam)
# ---------------------------------------------------------------------------

def mendelian_mating(parents, n_target, rng):
    """Random pairing, unlinked Mendelian transmission, **near-equal family size**.

    Each pair makes ``n_target // n_pairs`` offspring, and the remainder is
    distributed one extra to *randomly chosen* pairs — so family sizes differ by
    at most one and family-size variance is ~0. That minimal variance is the
    modelling choice behind `Ne ≈ 2N` (module docstring): it keeps drift weak so
    selection experiments are not swamped at the population sizes we can afford.
    Swap this for Poisson family sizes to recover `Ne ≈ N`.

    Two details Copilot flagged on PR #4, both fixed here rather than papered over:

    * When ``n_sel`` is odd one parent cannot pair. Because ``perm`` is shuffled,
      the unpaired parent is *random*, not systematically the last — but it is
      still dropped, which is the honest behaviour of an even-pairing scheme and
      is now stated rather than silent.
    * The remainder is spread to random pairs, **not** taken by truncating the
      offspring array from the end — the earlier ``[:n_target]`` slice
      systematically underweighted the last pair whenever ``n_target`` was not a
      multiple of ``n_pairs``. In the two configurations the gates actually use
      (drift: ``n_target = 2·n_pairs``; selection: an exact multiple) there was no
      remainder and the numbers are unaffected — but the general path was biased.
    """
    n_sel = parents.shape[1]
    if n_sel < 2:
        raise ValueError(f"need >= 2 parents to mate, got {n_sel}")
    perm = rng.permutation(n_sel)                       # random -> random drop if odd
    pairs = perm[: 2 * (n_sel // 2)].reshape(-1, 2)
    n_pairs = len(pairs)
    pA, pB = parents[:, pairs[:, 0]], parents[:, pairs[:, 1]]

    base, rem = divmod(n_target, n_pairs)
    blocks = []
    if base > 0:
        blocks.append(GEN.recombine(pA, pB, base, rng))
    if rem > 0:                                          # one extra to `rem` random pairs
        extra = rng.choice(n_pairs, size=rem, replace=False)
        blocks.append(GEN.recombine(pA[:, extra], pB[:, extra], 1, rng))
    return np.concatenate(blocks, axis=1)               # exactly n_target, no truncation


# ---------------------------------------------------------------------------
# Selection (a seam: "env or player", DESIGN.md §2E)
# ---------------------------------------------------------------------------

def truncation_toward(optimum, keep_fraction=0.5):
    """Keep the fraction closest to a fixed optimum shape. Milocco-Uller's."""
    def sel(z, rng):
        idx, _ = QG.truncation_select(z, optimum, keep_fraction)
        return idx
    return sel


def neutral_selection(keep_fraction=0.5):
    """No selection: a random subset reproduces.

    Not a no-op — it is the **null model**. Everything the loop does apart from
    selection still happens (development included, because organisms still
    develop whether or not it affects who breeds), so a neutral run isolates
    drift and is what gate #4a measures `Ne` from.
    """
    def sel(z, rng):
        n_keep = max(2, int(keep_fraction * len(z)))
        return rng.permutation(len(z))[:n_keep]
    return sel


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def heterozygosity(scores):
    """Mean expected heterozygosity `2pq` over loci, from realised frequencies.

    The standard drift observable: it decays geometrically at `1/(2Ne)` per
    generation under neutrality, which is what makes `Ne` measurable.
    """
    p, q = GEN.allele_frequencies(np.asarray(scores))
    return float(np.mean(2 * p * q))


def _quantgen_snapshot(scores, a, u, org, arch, basis, z, sel):
    """G, P, s, β and the breeder's prediction, for *this* generation.

    G is rebuilt from the sensitivity **at this generation's mean genome** and
    **this generation's realised allele frequencies** — both of which selection
    has been changing. That is the whole point of gate #4b: a G measured at
    generation 0 describes a population that no longer exists.
    """
    import jax.numpy as jnp
    from jax_morpho.evodevo import pipeline as PL

    mean_in = jnp.asarray(np.concatenate([a.mean(0), u.mean(0)]))
    J_tan = np.asarray(basis).T @ np.asarray(PL.phenotype_jacobian(mean_in, org))
    alpha = QG.average_effects(J_tan[:, :arch.n_genes], arch)
    G = QG.build_G_alleles(alpha, scores)

    P = np.asarray(QG.empirical_covariance(jnp.asarray(z)))
    s = z[sel].mean(0) - z.mean(0)
    beta = QG.selection_gradient(P, s)
    return dict(G=G, P=P, s=s, beta=beta, dz_pred=QG.lande_response(G, beta))


def evolve(scores, org, arch, *, n_generations, selection,
           variation=no_variation, mating=mendelian_mating, donors=None,
           n_env=2, sigma_env=0.02, seed=0, develop=True, basis=None,
           measure_response=False):
    """Run the loop, recording a per-generation history.

    ``develop=False`` skips development entirely and is **only** valid when
    selection ignores the phenotype (i.e. a neutral null). It exists because a
    drift measurement needs many replicate generations and no phenotype at all,
    and paying for development to throw it away would put the null model out of
    CI's reach. Passing it alongside phenotype-dependent selection is a
    programming error, so it is rejected rather than silently ignored.

    ``measure_response=True`` additionally rebuilds `G`, `P`, `s`, `β` and the
    breeder's prediction `Gβ` **every generation** — the instrumentation gate #4b
    reads. It costs one developmental Jacobian per generation.
    """
    rng = np.random.default_rng(seed)
    if develop and org is None:
        raise ValueError("develop=True needs an organism (org=None). Pass an "
                         "Organism, or set develop=False for a neutral drift run "
                         "whose selection ignores the phenotype.")
    if develop and basis is None:
        basis = PH.tangent_basis(org.ref)
    if measure_response and not develop:
        raise ValueError("measure_response=True requires develop=True — there is "
                         "no phenotype to measure a response in otherwise")

    hist = []
    for gen in range(n_generations):
        n = scores.shape[1]
        if develop:
            a = GEN.genome_from_scores(scores, arch)
            u = GEN.sample_environment(n, n_env, sigma_env, rng)
            z = RS.develop_tangent(a, u, org, basis)
        else:
            z = np.zeros((n, 1))            # placeholder; selection must ignore it

        sel = np.asarray(selection(z, rng))
        rec = dict(gen=gen, n=n, heterozygosity=heterozygosity(scores),
                   z_mean=z.mean(0).copy() if develop else None,
                   n_selected=len(sel))
        if measure_response:
            rec.update(_quantgen_snapshot(scores, a, u, org, arch, basis, z, sel))
        hist.append(rec)

        ctx = VariationContext(arch=arch, parents=scores[:, sel], donors=donors)
        offspring = mating(scores[:, sel], n, rng)
        scores = variation(offspring, ctx, rng)

    # final state, after the last round of reproduction
    hist.append(dict(gen=n_generations, n=scores.shape[1],
                     heterozygosity=heterozygosity(scores),
                     z_mean=None, n_selected=0))
    return scores, hist


def effective_population_size(H, drop_first=0):
    """`Ne` from a heterozygosity trajectory: `H_t = H_0 (1 - 1/(2Ne))^t`.

    Fits the geometric decay in log space. Returns `inf` if `H` does not decay
    (no drift), which is the honest answer rather than a division by zero.

    Fixation (`H → 0`, all loci homozygous) is guarded: `log(0) = −∞` would poison
    the fit. Any non-positive tail is dropped, and if fewer than two usable points
    remain the trajectory carried no fittable decay and the result is `inf`.
    """
    H = np.asarray(H, float)[drop_first:]
    good = np.isfinite(H) & (H > 0)
    if good.sum() < 2:
        return np.inf
    t = np.arange(len(H))[good]
    slope = np.polyfit(t, np.log(H[good]), 1)[0]
    rate = 1.0 - np.exp(slope)
    if rate <= 0:
        return np.inf
    return 1.0 / (2.0 * rate)
