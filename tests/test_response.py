"""Gate #3: the Fig-3C pattern reproduced with *our* development.

Phase 0 reproduced Milocco & Uller's Fig 3C on their toggle-switch ODE. Phase 3
asks whether the pattern survives replacing their ODE with a relaxing tissue and
their two traits with Procrustes shape:

    **a development-derived G predicts the response to selection; the phenotypic
    covariance P, used in its place, does not.**

What is and is not claimed
--------------------------
**Reproduced:** G predicts the realised response direction to a few degrees while
the naive P-based prediction is an order of magnitude worse — measured at every
allele frequency where the response is resolvable.

**Not reproduced: the monotone degradation of P as p → 0.** That tail of their
figure is exactly where our signal-to-noise dies. The realised response shrinks
with the genetic variance (∝ 2pq) while the noise on estimating it is set by the
environment-dominated phenotypic sd, which does not shrink — so SNR → 0 at low
MAF *intrinsically*. They buy that tail by brute force: 5000 individuals × 50
replays ≈ 5e5 developments per point, against ~1e4 here.

**And it cannot be bought by turning up the genetic variance.** Measured: raising
`sigma_gamma` from 0.02 to 0.08 lifts SNR but degrades `angle_G` from 3.3° to
18.8°, because larger genetic perturbations leave the linear regime G is defined
in. That is gate #2's small-perturbation constraint biting gate #3's measurement
— the two pull against each other, and the honest move is to sit where G is
accurate and admit the MAF range that costs us.

Every angle is therefore reported next to its SNR, and the gate is asserted only
where SNR > 3. Claiming the tail from a noise-dominated measurement would be
exactly the menu-fit §0 forbids.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_morpho.evodevo import genetics as GEN
from jax_morpho.evodevo import genome_map as GM
from jax_morpho.evodevo import phenotype as PH
from jax_morpho.evodevo import pipeline as PL
from jax_morpho.evodevo import quantgen as QG
from jax_morpho.evodevo import response as RS

N_GENES, N_ENV = 4, 2
SNR_FLOOR = 3.0


@pytest.fixture(scope="module")
def setup():
    grn = GM.init_grn(jax.random.key(0), N_GENES + N_ENV, hidden=16, scale=1.5)
    # landmark_stride=5 -> k=4 landmarks -> tangent dim 2k-4 = 4, so P is
    # invertible with 4 genes + 2 environmental inputs feeding it.
    org = PL.make_organism(grn, jnp.zeros(N_GENES + N_ENV), n_rings=2,
                           landmark_stride=5)
    arch = GEN.make_architecture(N_GENES, loci_per_gene=5, sigma_gamma=0.02,
                                 rng=np.random.default_rng(0))
    return dict(org=org, arch=arch,
                optimum=RS.reference_optimum(org, arch))


#: Gated MAF values. Chosen for resolvability, not for a flattering answer: at
#: p <= 0.125 the response falls under the noise floor at any sample size CI can
#: afford (see the module docstring), and a noise-dominated angle asserts
#: nothing. `test_the_measurement_is_actually_resolved` enforces the choice
#: rather than trusting it; `examples/demo_phase3_gate.py` shows the full sweep
#: including the points where the measurement dies.
GATED_P = [0.5, 0.25]


@pytest.fixture(scope="module")
def sweep(setup):
    return [RS.simulate_response(p, setup["org"], setup["arch"],
                                 optimum=setup["optimum"], n_ind=700,
                                 n_replays=8, seed=k)
            for k, p in enumerate(GATED_P)]


# ---------------------------------------------------------------------------
# Gate #3
# ---------------------------------------------------------------------------

class TestGate3:
    def test_the_measurement_is_actually_resolved(self, sweep):
        """Guard the guard: if SNR collapsed, every angle below is noise and the
        gate would be asserting nothing."""
        for r in sweep:
            assert r["snr"] > SNR_FLOOR, (
                f"p={r['p']}: response below the noise floor (snr={r['snr']:.1f}); "
                "the angles are meaningless — raise n_ind/n_replays")

    def test_G_predicts_the_response_to_within_the_noise_floor(self, sweep):
        """The gate — stated against the measurement's own resolution rather
        than an arbitrary threshold.

        A response estimated at signal-to-noise `snr` can be tilted by up to
        `arcsin(1/snr)` by noise alone, so no prediction can be *shown* to align
        better than that. G sits at that floor: measured `angle_G` tracks
        `arcsin(1/snr)` across the whole range (snr 5 → 15.7° observed vs 11.5°
        floor; snr 20 → 1.6° vs 2.8°). **G's error is consistent with zero** —
        what is left is what we cannot resolve, not what G got wrong.
        """
        for r in sweep:
            assert r["angle_G"] < 2.5 * r["noise_angle"], (
                f"p={r['p']}: G off by {r['angle_G']:.1f}°, well beyond the "
                f"{r['noise_angle']:.1f}° noise floor (snr={r['snr']:.1f}) — "
                "that would be a real systematic error, not a measurement limit")

    def test_P_is_wrong_beyond_any_noise_excuse(self, sweep):
        """...and the contrast, which is the actual claim. P's error is *not*
        at the noise floor — it is an order of magnitude above it, so using P
        where G belongs is a genuine systematic error. Measured: 20°–114°
        against a floor of ~3–5°."""
        for r in sweep:
            assert r["angle_P"] > 4.0 * r["noise_angle"], (
                f"p={r['p']}: P's {r['angle_P']:.1f}° is within noise "
                f"({r['noise_angle']:.1f}°) — no contrast to report")
            assert r["angle_P"] > 3.0 * r["angle_G"], (
                f"p={r['p']}: no G-vs-P contrast "
                f"(G {r['angle_G']:.1f}° vs P {r['angle_P']:.1f}°)")
        assert max(r["angle_P"] for r in sweep) > 20.0

    def test_heritability_is_neither_zero_nor_one(self, sweep):
        """Guard the regime: h²=0 means no response to predict, h²=1 means P=G
        and the comparison is vacuous. The Fig-3C question only exists in
        between."""
        for r in sweep:
            assert 0.05 < r["heritability"] < 0.95

    def test_response_shrinks_with_the_genetic_variance(self, sweep):
        """Sanity on the mechanism: as the swept genes go rare, 2pq falls and so
        must the realised response. If it didn't, we'd be measuring drift."""
        mags = [float(np.linalg.norm(r["dz_obs"])) for r in sweep]
        assert mags[0] > mags[-1]

    def test_P_is_invertible_in_tangent_coordinates(self, sweep):
        """Why the whole thing is well posed — β = P⁻¹s is meaningless in ambient
        shape coordinates, where P is singular by construction."""
        for r in sweep:
            assert r["rank_P"] == r["tangent_dim"] == 4


class TestTangentSpace:
    def test_basis_annihilates_the_four_constraints(self, setup):
        ref = setup["org"].ref
        B = PH.tangent_basis(ref)
        z0 = jnp.asarray(ref).ravel()
        k = z0.shape[0] // 2
        xy = z0.reshape(k, 2)
        tx = jnp.stack([jnp.ones(k), jnp.zeros(k)], -1).ravel()
        ty = jnp.stack([jnp.zeros(k), jnp.ones(k)], -1).ravel()
        rot = jnp.stack([-xy[:, 1], xy[:, 0]], -1).ravel()
        for v in (tx, ty, rot, z0):                    # translations, rotation, scale
            assert float(jnp.abs(B.T @ v).max()) < 1e-10
        assert B.shape == (2 * k, 2 * k - 4)
        assert float(jnp.abs(B.T @ B - jnp.eye(2 * k - 4)).max()) < 1e-10

    def test_tangent_projection_makes_P_full_rank(self, setup):
        """Ambient P is singular; tangent P is not. The entire reason the
        quantitative genetics can be done at all."""
        org = setup["org"]
        rng = np.random.default_rng(0)
        A = np.concatenate([rng.normal(0, 0.01, (200, N_GENES)),
                            rng.normal(0, 0.02, (200, N_ENV))], 1)
        Z = PL.phenotype_population(jnp.asarray(A), org)
        zt = np.asarray(PH.tangent_coords(Z, org.ref))
        amb = np.asarray(QG.empirical_covariance(Z))
        tan = np.asarray(QG.empirical_covariance(jnp.asarray(zt)))
        assert np.linalg.matrix_rank(amb, tol=1e-14) < amb.shape[0]   # singular
        assert np.linalg.matrix_rank(tan, tol=1e-14) == tan.shape[0]  # invertible


# ---------------------------------------------------------------------------
# The reverse-mode path (DESIGN.md §2D)
# ---------------------------------------------------------------------------

class TestResponseVJP:
    def test_vjp_matches_the_dense_jacobian(self, setup):
        """``Jᵀβ`` in one solve must equal ``βᵀJ`` from the formed Jacobian."""
        org = setup["org"]
        a = jnp.zeros(N_GENES + N_ENV)
        beta = jnp.asarray(np.random.default_rng(3).normal(
            0, 1, 2 * org.idx.shape[0]))
        got = np.asarray(PL.phenotype_vjp(a, org, beta))
        want = np.asarray(beta) @ np.asarray(PL.phenotype_jacobian(a, org))
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-6

    def test_jvp_matches_the_dense_jacobian(self, setup):
        """...and the forward twin, ``J δa``."""
        org = setup["org"]
        a = jnp.zeros(N_GENES + N_ENV)
        da = jnp.asarray(np.random.default_rng(4).normal(0, 1, N_GENES + N_ENV))
        got = np.asarray(PL.phenotype_jvp(a, org, da))
        want = np.asarray(PL.phenotype_jacobian(a, org)) @ np.asarray(da)
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-6

    def test_lande_response_without_forming_G_or_J(self, setup):
        """``Δz̄ = J M Jᵀβ`` in **two** solves, matching the explicitly-formed
        answer.

        The path that scales: forming J costs one solve per gene; this costs one
        each way, independent of gene *and* trait count. At organism scale a
        solve is a matrix-free CG over ~10⁶ unknowns.
        """
        org = setup["org"]
        a = jnp.zeros(N_GENES + N_ENV)
        rng = np.random.default_rng(3)
        M = jnp.asarray(np.diag(rng.uniform(0.5, 2.0, N_GENES + N_ENV)))
        beta = jnp.asarray(rng.normal(0, 1, 2 * org.idx.shape[0]))

        got = PL.lande_response_vjp(a, org, beta, M)
        J = PL.phenotype_jacobian(a, org)
        want = J @ (M @ (J.T @ beta))
        rel = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert rel < 1e-6, f"VJP path disagrees with J M Jᵀβ: {rel:.3e}"

    def test_naive_reverse_mode_through_the_solver_does_not_work(self, setup):
        """Pin *why* the implicit transpose is necessary rather than tidy.

        `jax.vjp` through the genome→phenotype map hits `equilibrate`'s
        `lax.while_loop` and raises. Unrolling it to a `scan` would make it run —
        and return the wrong sensitivity, per Phase 1. If this ever stops
        raising, someone made the solver reverse-differentiable and the Phase 1
        argument needs revisiting *before* anyone trusts the gradient.
        """
        org = setup["org"]
        a = jnp.zeros(N_GENES + N_ENV)
        z, pullback = jax.vjp(lambda g: PL.phenotype(g, org), a)   # lazy: fine
        with pytest.raises(ValueError, match="[Rr]everse-mode"):
            pullback(jnp.ones_like(z))                             # this is where it dies


# ---------------------------------------------------------------------------
# Genetics
# ---------------------------------------------------------------------------

class TestGenetics:
    def test_allele_frequencies_recover_the_sampling_p(self):
        rng = np.random.default_rng(0)
        for p in (0.5, 0.1):
            s = GEN.sample_genotypes(p, 40, 4000, rng)
            p_hat, q_hat = GEN.allele_frequencies(s)
            assert abs(p_hat.mean() - p) < 0.02
            assert np.allclose(p_hat + q_hat, 1.0)

    def test_genome_is_additive_over_a_genes_loci(self):
        arch = GEN.make_architecture(3, 4, 0.1, np.random.default_rng(1))
        scores = GEN.sample_genotypes(0.5, arch.n_loci, 50, np.random.default_rng(2))
        a = GEN.genome_from_scores(scores, arch)
        assert a.shape == (50, 3)
        # gene 0 is exactly the sum over its own loci, and nothing else's
        loci0 = np.where(arch.gene_of_locus == 0)[0]
        expect = (scores[loci0] * arch.gamma[loci0][:, None]).sum(0)
        assert np.abs(a[:, 0] - expect).max() < 1e-12

    def test_recombination_preserves_the_allele_alphabet(self):
        rng = np.random.default_rng(0)
        s = GEN.sample_genotypes(0.5, 10, 40, rng)
        off = GEN.recombine(s[:, :20], s[:, 20:], 2, rng)
        assert off.shape == (10, 40)
        assert set(np.unique(off)).issubset({-1, 0, 1})

    def test_environment_is_not_heritable_but_is_present(self):
        """Zero environmental variance would collapse P onto G and make Fig 3C
        vacuous — the contrast being tested would not exist."""
        u = GEN.sample_environment(1000, 2, 0.05, np.random.default_rng(0))
        assert u.shape == (1000, 2)
        assert 0.03 < u.std() < 0.07
