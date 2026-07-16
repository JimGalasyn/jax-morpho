"""Phase 3: the one-generation response to selection, on *our* development.

This closes `genotype → development → phenotype → selection → response` with the
mechanical engine in the middle, and asks the question Milocco & Uller's Fig 3C
asks:

    **Does a development-derived G predict the response to selection where the
    phenotypic covariance P does not?**

Their answer, on a toggle-switch ODE with two traits: yes — G stays aligned with
the realised response across allele frequencies while P misaligns badly at low
minor-allele frequency. Phase 0 reproduced that on their model
(`reference_mu`). Phase 3 asks whether the *pattern* survives when their ODE is
replaced by a relaxing tissue and their two traits by Procrustes shape.

The protocol, following theirs
------------------------------
1. Sample diploid genotypes at minor-allele frequency `p`; additive allelic
   effects build the genome vector.
2. Develop each individual (genome **+ non-heritable environment**) to its
   equilibrium form, and read out Procrustes shape in **tangent coordinates**.
3. `G` from *our* sensitivity: `α_l = γ_l · s_l` with `s` from implicit-diff at
   the population mean, then `G = Σ 2p_l q_l α_l α_lᵀ` (their Eq. 12).
4. `P = cov(z)`; truncation-select the half closest to an optimum shape; the
   selection differential is `s = z̄_sel − z̄`.
5. Predict two ways: `Δz̄ = Gβ` with `β = P⁻¹s` (Lande), versus the naive `Δz̄ = s`
   (which is what using P *as* G amounts to).
6. Recombine the selected parents, **develop the offspring**, and measure the
   realised `Δz̄`. Compare directions.

Why each piece is not optional
------------------------------
* **Environment.** With no non-heritable variance, `P = G` exactly and the
  comparison is vacuous — the two predictions coincide. Their `u` exists for the
  same reason.
* **Tangent coordinates.** `β = P⁻¹s` in ambient shape coordinates is
  meaningless: P is singular by construction there (2k coordinates, 2k−4 degrees
  of freedom). See :func:`~jax_morpho.evodevo.phenotype.tangent_basis`.
* **Developing the offspring.** The realised response has to come from the
  engine, not from the same linearisation being tested — otherwise the gate
  confirms itself.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jax_morpho.evodevo import genetics as GEN
from jax_morpho.evodevo import phenotype as PH
from jax_morpho.evodevo import pipeline as PL
from jax_morpho.evodevo import quantgen as QG


def _inputs(a, u):
    """Assemble the developmental map's input: heritable genome + environment."""
    return np.concatenate([np.asarray(a), np.asarray(u)], axis=-1)


def develop_tangent(a, u, org, basis):
    """Genomes + environments → shapes in tangent coordinates. Batched.

    The genome/environment split is a convention between caller and organism —
    the GRN just sees one input vector — so a mismatch is checked here rather
    than surfacing as a shape error deep inside ``grn_field``.
    """
    full = jnp.asarray(_inputs(a, u))
    expected = org.grn.W1.shape[0] - 2          # less the (x, y) positional input
    if full.shape[-1] != expected:
        raise ValueError(
            f"genome ({np.shape(a)[-1]}) + environment ({np.shape(u)[-1]}) = "
            f"{full.shape[-1]} inputs, but this organism's GRN expects "
            f"{expected} (its W1 takes {org.grn.W1.shape[0]}, less 2 positional)")
    Z = PL.phenotype_population(full, org)
    return np.asarray(PH.tangent_coords(Z, org.ref, basis))


def simulate_response(p, org, arch, *, n_env=2, sigma_env=0.02, n_ind=600,
                      optimum=None, optimum_scale=1.0, n_replays=8, seed=0,
                      swept_genes=None, p_fixed=0.5):
    """One point of the Fig-3C sweep, on our mechanical development.

    Returns a dict with `G` (sensitivity-derived), `P`, the selection
    differential `s`, the two predictions, the realised response, and the angles
    between them.

    The anisotropy is the point
    ---------------------------
    `p` is applied only to `swept_genes` (default: the first half); the rest stay
    at `p_fixed = 0.5`. That asymmetry is Milocco & Uller's protocol — they hold
    θ₁ at 0.5 and sweep θ₂ — and it is *the mechanism*, not a detail. Lowering
    every gene's frequency together merely shrinks G uniformly; G stays full rank,
    the response stays parallel to the selection differential, and **no
    G-versus-P contrast appears** (measured: `angle_P` erratic, no trend).
    Sweeping only some genes collapses the variance along *those* directions, so
    G becomes rank-deficient **relative to** P: the realised response is confined
    to the subspace development can still vary, while the raw selection
    differential `s` is not. That is what P gets wrong and G gets right.

    The optimum must be FIXED across the sweep
    ------------------------------------------
    Milocco & Uller select toward a fixed point (their (4,4)). Recomputing the
    optimum per population — e.g. `z̄ + scale·z.std()` — silently destroys the
    experiment: selection then always pulls *along whatever variance the
    population happens to have*, so `s` stays inside the responsive subspace and
    can never misalign, no matter how far the swept genes' frequency falls
    (measured: `angle_P` flat at ~15–28° with no trend). Pass `optimum` from
    :func:`run_sweep`, which derives it once at `p_fixed`. `optimum_scale` only
    sets how far that fixed target sits from the reference mean, in phenotypic SD.
    """
    rng = np.random.default_rng(seed)
    basis = PH.tangent_basis(org.ref)

    # -- 1. genotypes -> genomes, plus non-heritable environment ------------
    if swept_genes is None:
        swept_genes = np.arange(arch.n_genes // 2)
    swept = np.isin(arch.gene_of_locus, np.asarray(swept_genes))    # (n_loci,)
    scores = np.where(
        swept[:, None],
        GEN.sample_genotypes(p, arch.n_loci, n_ind, rng),
        GEN.sample_genotypes(p_fixed, arch.n_loci, n_ind, rng))
    a = GEN.genome_from_scores(scores, arch)                  # (n_ind, n_genes)
    u = GEN.sample_environment(n_ind, n_env, sigma_env, rng)

    # -- 2. develop the population -----------------------------------------
    z = develop_tangent(a, u, org, basis)                     # (n_ind, 2k-4)

    # -- 3. G from OUR sensitivity, at the population mean ------------------
    mean_in = jnp.asarray(_inputs(a.mean(0), u.mean(0)))
    J_amb = np.asarray(PL.phenotype_jacobian(mean_in, org))   # (2k, n_genes+n_env)
    J_tan = np.asarray(basis).T @ J_amb                       # (2k-4, ...)
    J_gen = J_tan[:, :arch.n_genes]                           # heritable part only
    alpha = QG.average_effects(J_gen, arch)                   # (n_loci, 2k-4)
    G = QG.build_G_alleles(alpha, scores)

    # -- 4. P, selection, the differential ---------------------------------
    P = np.asarray(QG.empirical_covariance(jnp.asarray(z)))
    if optimum is None:      # standalone use; run_sweep supplies a fixed one
        optimum = z.mean(0) + optimum_scale * z.std(0)
    sel, s = QG.truncation_select(z, np.asarray(optimum), keep_fraction=0.5)

    # -- 5. the two predictions --------------------------------------------
    beta = QG.selection_gradient(P, s)
    dz_lande = QG.lande_response(G, beta)
    dz_naive = s.copy()          # using P as if it were G

    # -- 6. the realised response: recombine, develop the offspring ---------
    parents = scores[:, sel]
    n_sel = len(sel)
    obs, n_off_total = [], 0
    for _ in range(n_replays):
        perm = rng.permutation(n_sel)
        pairs = perm[: 2 * (n_sel // 2)].reshape(-1, 2)
        off = GEN.recombine(parents[:, pairs[:, 0]], parents[:, pairs[:, 1]],
                            2, rng)
        a_off = GEN.genome_from_scores(off, arch)
        u_off = GEN.sample_environment(a_off.shape[0], n_env, sigma_env, rng)
        z_off = develop_tangent(a_off, u_off, org, basis)
        obs.append(z_off.mean(0) - z.mean(0))
        n_off_total += z_off.shape[0]
    dz_obs = np.mean(obs, axis=0)

    # -- 7. is the response even measurable? -------------------------------
    # The realised response shrinks with the genetic variance (∝ 2pq), but the
    # noise on estimating it is set by the *phenotypic* sd — which is
    # environment-dominated and does not shrink. So SNR → 0 at low allele
    # frequency **intrinsically**, for any parameter choice. Below snr ≈ 3 the
    # measured angles are noise and mean nothing; report it rather than let a
    # reader mistake scatter for a result. (Milocco & Uller reach p ~ 1e-3 by
    # brute force: 5000 individuals x 50 replays ≈ 5e5 developments per point.)
    noise = np.sqrt(np.trace(P) / P.shape[0]) / np.sqrt(max(n_off_total, 1))
    snr = float(np.linalg.norm(dz_obs) / max(noise, 1e-300))

    # The angular resolution the measurement itself has. A noise vector of
    # relative size 1/snr can tilt the measured response by up to arcsin(1/snr),
    # so *no* prediction — however perfect — can be shown to align better than
    # this. It is the yardstick the angles below must be read against: an angle
    # at the noise floor means "consistent with exact", not "1.8° wrong".
    noise_angle = float(np.degrees(np.arcsin(min(1.0 / max(snr, 1e-300), 1.0))))

    return dict(p=p, G=G, P=P, s=s, beta=beta,
                dz_lande=dz_lande, dz_naive=dz_naive, dz_obs=dz_obs,
                angle_G=QG.angle_deg(dz_obs, dz_lande),
                angle_P=QG.angle_deg(dz_obs, dz_naive),
                snr=snr, noise_angle=noise_angle, n_offspring=n_off_total,
                heritability=float(np.trace(G) / np.trace(P)),
                rank_P=int(np.linalg.matrix_rank(P, tol=1e-12)),
                tangent_dim=int(basis.shape[1]))


def reference_optimum(org, arch, *, n_env=2, sigma_env=0.02, n_ind=400,
                      optimum_scale=1.0, p_fixed=0.5, seed=99):
    """A selective optimum fixed once, from an unswept (p = 0.5) population.

    Held constant across the whole sweep — see :func:`simulate_response` on why
    an adaptive optimum destroys the experiment.
    """
    rng = np.random.default_rng(seed)
    basis = PH.tangent_basis(org.ref)
    scores = GEN.sample_genotypes(p_fixed, arch.n_loci, n_ind, rng)
    a = GEN.genome_from_scores(scores, arch)
    u = GEN.sample_environment(n_ind, n_env, sigma_env, rng)
    z = develop_tangent(a, u, org, basis)
    return z.mean(0) + optimum_scale * z.std(0)


def run_sweep(org, arch, p_values=None, **kw):
    """Sweep the minor-allele frequency — the Fig-3C x-axis.

    Their sweep is `0.5 / 2ᵏ`. The pattern to look for: `angle_G` stays small
    while `angle_P` grows as `p` falls and the swept genes' variance collapses.
    The optimum is fixed once here and reused at every `p`.
    """
    if p_values is None:
        p_values = 0.5 / 2.0 ** np.arange(0, 6)
    opt_kw = {k: kw[k] for k in ("n_env", "sigma_env", "optimum_scale", "p_fixed")
              if k in kw}
    opt = reference_optimum(org, arch, **opt_kw)
    return [simulate_response(float(p), org, arch, optimum=opt, seed=k, **kw)
            for k, p in enumerate(p_values)]
