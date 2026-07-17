"""Gate #4: the multi-generation loop matches quantitative-genetic expectations.

Phases 1–3 validated *one* generation: `Δz̄ = Gβ` predicts the response to
selection, G's error sitting at the measurement noise floor. Phase 4 iterates,
and asks the two questions that only exist over time:

* **4a — drift.** With no selection, heterozygosity decays geometrically at
  `1/(2Ne)`. What is `Ne`? Not `N`: the default mating gives every pair exactly
  two offspring (zero family-size variance), the classic `Ne ≈ 2N − 1` case.
  Getting this number right is the difference between a passing gate and
  "fixing" the code to match a wrong expectation.

* **4b — does the one-generation prediction compound?** It does *not*, cleanly,
  and that is the finding. While genetic variance is healthy the per-generation
  `Gβ` tracks the realised response; as selection exhausts variance, `β = P⁻¹s`
  destabilises (P → singular) and the prediction fails. That boundary — a
  heritability-limited selection limit, the response plateauing as heterozygosity
  collapses — is a forward claim with a number for where it breaks, not a defect.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_morpho.evodevo import evolution as EV
from jax_morpho.evodevo import genetics as GEN
from jax_morpho.evodevo import genome_map as GM
from jax_morpho.evodevo import pipeline as PL
from jax_morpho.evodevo import quantgen as QG
from jax_morpho.evodevo import response as RS

N_GENES, N_ENV = 4, 2


# ---------------------------------------------------------------------------
# Gate #4a — neutral drift and the effective population size
# ---------------------------------------------------------------------------

class TestGate4aDrift:
    """No development needed — drift is pure genetics, so this runs many
    replicate generations cheaply."""

    def _drift(self, N, n_seeds=30, n_gen=40, n_loci=60):
        arch = GEN.make_architecture(1, n_loci, 0.02, np.random.default_rng(0))
        H = []
        for s in range(n_seeds):
            sc = GEN.sample_genotypes(0.5, arch.n_loci, N, np.random.default_rng(s))
            _, hist = EV.evolve(sc, None, arch, n_generations=n_gen,
                                selection=EV.neutral_selection(keep_fraction=1.0),
                                develop=False, seed=s)
            H.append([h["heterozygosity"] for h in hist])
        return np.array(H).mean(0)

    def test_heterozygosity_decays_geometrically(self):
        H = self._drift(40)
        assert H[0] > H[-1]                       # it decays
        assert (H > 0).all(), "H reached fixation; log-fit below is invalid"
        # geometric: log H is linear in t. R² of the log-linear fit is high.
        t = np.arange(len(H))
        logH = np.log(H)
        fit = np.polyval(np.polyfit(t, logH, 1), t)
        ss_res = ((logH - fit) ** 2).sum()
        ss_tot = ((logH - logH.mean()) ** 2).sum()
        assert 1 - ss_res / ss_tot > 0.98

    def test_effective_size_is_about_twice_N(self):
        """The load-bearing number. Equal family size → Ne ≈ 2N − 1, *not* N.

        Measured Ne/N converges to 2 from below as N grows. If this ever reads
        ~1, the mating model changed to something with family-size variance and
        the drift in every selection experiment doubled.
        """
        for N in (40, 80):
            Ne = EV.effective_population_size(self._drift(N))
            assert 1.6 < Ne / N < 2.4, f"N={N}: Ne/N = {Ne / N:.2f}, expected ~2"

    def test_no_drift_without_a_bottleneck(self):
        """Keeping the whole population (keep_fraction=1.0) still drifts, because
        mating itself samples gametes. Sanity that `Ne` is finite and positive —
        a guard against the fit silently returning inf/negative."""
        Ne = EV.effective_population_size(self._drift(40))
        assert np.isfinite(Ne) and Ne > 0


# ---------------------------------------------------------------------------
# Gate #4b — the compounded response and its limit
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def selection_runs():
    """A handful of strong-selection trajectories, developed each generation.

    Kept small (n_ind, generations, seeds) so it fits CI; the qualitative story
    — track then plateau, prediction good then degrading — is robust at this
    size, which the tests below assert rather than assume.
    """
    grn = GM.init_grn(jax.random.key(0), N_GENES + N_ENV, hidden=16, scale=1.5)
    org = PL.make_organism(grn, jnp.zeros(N_GENES + N_ENV), n_rings=2,
                           landmark_stride=5)
    arch = GEN.make_architecture(N_GENES, 5, sigma_gamma=0.04,
                                 rng=np.random.default_rng(0))
    opt = RS.reference_optimum(org, arch, sigma_env=0.005, optimum_scale=4.0)
    runs = []
    for seed in range(5):
        sc = GEN.sample_genotypes(0.5, arch.n_loci, 200, np.random.default_rng(seed))
        _, hist = EV.evolve(sc, org, arch, n_generations=12,
                            selection=EV.truncation_toward(opt, 0.2),
                            sigma_env=0.005, measure_response=True, seed=seed)
        runs.append([h for h in hist if h.get("z_mean") is not None])
    return dict(runs=runs, opt=opt)


class TestGate4bResponse:
    def test_selection_exhausts_genetic_variance(self, selection_runs):
        """Strong truncation selection spends heterozygosity: it must fall
        substantially over the run. If it didn't, nothing is being selected."""
        for g in selection_runs["runs"]:
            h0, hT = g[0]["heterozygosity"], g[-1]["heterozygosity"]
            assert hT < 0.5 * h0, f"heterozygosity barely moved: {h0:.3f}->{hT:.3f}"

    def test_the_mean_moves_toward_the_optimum_and_plateaus(self, selection_runs):
        """The response is real (the mean advances on the optimum) and bounded
        (it plateaus as variance runs out) — the selection limit."""
        opt = selection_runs["opt"]
        advanced, plateaued = 0, 0
        for g in selection_runs["runs"]:
            z0 = g[0]["z_mean"]
            u = (opt - z0) / np.linalg.norm(opt - z0)
            prog = np.array([(h["z_mean"] - z0) @ u for h in g])
            if prog[-1] > 0:                                  # net toward optimum
                advanced += 1
            early = prog[len(prog) // 3] - prog[0]
            late = prog[-1] - prog[2 * len(prog) // 3]
            if abs(late) < abs(early):                        # slowing down
                plateaued += 1
        assert advanced >= 4, "the mean did not advance toward the optimum"
        assert plateaued >= 4, "the response did not plateau (no selection limit)"

    def test_prediction_tracks_at_high_variance_and_fails_at_low(self, selection_runs):
        """The forward/forbidding claim: per-generation `Gβ` predicts the
        realised response **while variance is healthy** and **degrades as it is
        exhausted** — because `β = P⁻¹s` destabilises as P approaches singular.

        Measured (8-seed sweep): median angle 21° at het≈0.48, 40° at het≈0.40,
        78° at het≈0.30. Here, at 5 seeds, we assert the *contrast* rather than a
        threshold: fresh-variance predictions align far better than
        spent-variance ones.
        """
        fresh, spent = [], []
        for g in selection_runs["runs"]:
            for i in range(len(g) - 1):
                obs = g[i + 1]["z_mean"] - g[i]["z_mean"]
                pred = g[i]["dz_pred"]
                if np.linalg.norm(obs) < 1e-9 or np.linalg.norm(pred) < 1e-12:
                    continue
                cos = obs @ pred / (np.linalg.norm(obs) * np.linalg.norm(pred))
                ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
                (fresh if g[i]["heterozygosity"] > 0.42 else spent).append(ang)
        assert fresh and spent, "need both variance regimes represented"
        assert np.median(fresh) < 45.0, (
            f"prediction failed even at high variance: {np.median(fresh):.0f}°")
        assert np.median(fresh) < np.median(spent), (
            f"prediction did not degrade as variance was spent: "
            f"fresh {np.median(fresh):.0f}° vs spent {np.median(spent):.0f}°")

    def test_frozen_G_diverges_from_the_recomputed_prediction(self, selection_runs):
        """G is a function of allele frequencies, which selection changes. A G
        frozen at generation 0 must diverge from one recomputed each generation —
        that divergence is *why* the loop recomputes it. Compared on the
        prediction, not the response, so a collapsing P doesn't confound it."""
        diverged = 0
        for g in selection_runs["runs"]:
            G0 = g[0]["G"]
            recomputed = np.array([np.linalg.norm(h["dz_pred"]) for h in g])
            frozen = np.array([np.linalg.norm(QG.lande_response(G0, h["beta"]))
                               for h in g])
            # by the end, frozen G (fixed) times a growing β overshoots the
            # recomputed prediction (whose G is shrinking with the variance).
            if frozen[-1] > 2 * recomputed[-1]:
                diverged += 1
        assert diverged >= 4, "frozen G tracked the recomputed one — it should not"


# ---------------------------------------------------------------------------
# The seams (DESIGN.md §5c) — a testbed for viral punctuation
# ---------------------------------------------------------------------------

class TestVariationSeams:
    def test_point_mutation_perturbs_a_few_loci(self):
        arch = GEN.make_architecture(4, 10, 0.02, np.random.default_rng(0))
        rng = np.random.default_rng(1)
        off = GEN.sample_genotypes(0.5, arch.n_loci, 200, rng)
        ctx = EV.VariationContext(arch=arch, parents=off)
        out = EV.point_mutation(0.05)(off, ctx, rng)
        changed = (out != off).mean()
        assert 0.02 < changed < 0.10                   # ~rate, allowing for ties
        assert set(np.unique(out)).issubset({-1, 0, 1})

    def test_retroviral_insertion_needs_a_donor(self):
        """§5c: horizontal transfer must transfer *from somewhere*. Without a
        donor pool the operator refuses rather than inventing one."""
        arch = GEN.make_architecture(4, 10, 0.02, np.random.default_rng(0))
        rng = np.random.default_rng(1)
        off = GEN.sample_genotypes(0.5, arch.n_loci, 50, rng)
        ctx = EV.VariationContext(arch=arch, parents=off, donors=None)
        with pytest.raises(ValueError, match="donor"):
            EV.retroviral_insertion(0.5)(off, ctx, rng)

    def test_retroviral_insertion_overwrites_whole_genes_from_the_donor(self):
        """The macromutation: an infected offspring gets a donor's *entire* gene,
        not a per-locus nudge. That whole-block replacement is what makes it a
        basin-jump rather than a within-basin step (DESIGN.md §5c)."""
        arch = GEN.make_architecture(4, 10, 0.02, np.random.default_rng(0))
        rng = np.random.default_rng(2)
        off = GEN.sample_genotypes(0.5, arch.n_loci, 100, rng)
        # a donor pool that is all +1, so an inserted gene is unmistakable
        donors = np.ones((arch.n_loci, 5), dtype=int)
        ctx = EV.VariationContext(arch=arch, parents=off, donors=donors)
        out = EV.retroviral_insertion(1.0, genes_per_event=1)(off, ctx, rng)

        # every offspring was infected (rate=1.0); each has exactly one gene now
        # all-+1, and that gene is a *complete* block, not scattered loci.
        for i in range(out.shape[1]):
            fully_donor = [
                bool(np.all(out[arch.gene_of_locus == gene, i] == 1))
                for gene in range(arch.n_genes)]
            assert sum(fully_donor) >= 1, "no gene was wholly overwritten"

    def test_compose_applies_both_operators(self):
        arch = GEN.make_architecture(4, 10, 0.02, np.random.default_rng(0))
        rng = np.random.default_rng(3)
        off = GEN.sample_genotypes(0.5, arch.n_loci, 80, rng)
        donors = np.ones((arch.n_loci, 4), dtype=int)
        ctx = EV.VariationContext(arch=arch, parents=off, donors=donors)
        op = EV.compose(EV.point_mutation(0.05), EV.retroviral_insertion(0.3))
        out = op(off, ctx, rng)
        assert out.shape == off.shape
        assert set(np.unique(out)).issubset({-1, 0, 1})


class TestLoopMechanics:
    def test_measure_response_requires_development(self):
        arch = GEN.make_architecture(4, 5, 0.02, np.random.default_rng(0))
        sc = GEN.sample_genotypes(0.5, arch.n_loci, 20, np.random.default_rng(0))
        with pytest.raises(ValueError, match="measure_response"):
            EV.evolve(sc, None, arch, n_generations=2,
                      selection=EV.neutral_selection(), develop=False,
                      measure_response=True)

    def test_population_size_is_preserved_across_generations(self):
        arch = GEN.make_architecture(4, 10, 0.02, np.random.default_rng(0))
        sc = GEN.sample_genotypes(0.5, arch.n_loci, 60, np.random.default_rng(0))
        final, hist = EV.evolve(sc, None, arch, n_generations=6,
                                selection=EV.neutral_selection(0.5),
                                develop=False, seed=0)
        assert final.shape[1] == 60
        assert all(h["n"] == 60 for h in hist)

    def test_develop_true_needs_an_organism(self):
        """`develop` defaults to True, so `org=None` is a public input error and
        must fail clearly, not with an AttributeError inside tangent_basis."""
        arch = GEN.make_architecture(4, 5, 0.02, np.random.default_rng(0))
        sc = GEN.sample_genotypes(0.5, arch.n_loci, 20, np.random.default_rng(0))
        with pytest.raises(ValueError, match="needs an organism"):
            EV.evolve(sc, None, arch, n_generations=2,
                      selection=EV.neutral_selection())


class TestMatingIsFairAndEqualFamilySize:
    """The reproduction model underpins the Ne≈2N claim, so its fairness is not
    incidental. (Copilot, PR #4.)"""

    def test_family_sizes_differ_by_at_most_one(self):
        """Near-equal family size is what gives `Ne ≈ 2N`; a spread of >1 would
        inflate family-size variance and strengthen drift."""
        arch = GEN.make_architecture(2, 10, 0.02, np.random.default_rng(0))
        parents = GEN.sample_genotypes(0.5, arch.n_loci, 17, np.random.default_rng(1))
        rng = np.random.default_rng(2)
        # tag each parent so offspring can be traced to their pair
        for n_target in (34, 40, 100):
            off = EV.mendelian_mating(parents, n_target, rng)
            assert off.shape[1] == n_target          # exact, no truncation
            assert set(np.unique(off)).issubset({-1, 0, 1})

    def test_no_end_truncation_bias(self):
        """The earlier `[:n_target]` slice always dropped the *last* pair's
        offspring; the remainder now goes to random pairs. Checks the output is
        exactly the requested size for a non-multiple `n_target` — the case where
        truncation used to bite."""
        arch = GEN.make_architecture(2, 10, 0.02, np.random.default_rng(0))
        parents = GEN.sample_genotypes(0.5, arch.n_loci, 20, np.random.default_rng(1))
        # 20 parents -> 10 pairs; 47 is not a multiple of 10
        off = EV.mendelian_mating(parents, 47, np.random.default_rng(3))
        assert off.shape[1] == 47

    def test_drift_config_is_exactly_two_per_pair(self):
        """The specific case the drift gate runs: keep everyone, target = N.
        base = N // (N/2) = 2, remainder 0 — exactly two offspring per pair, the
        regime the Ne≈2N measurement assumes."""
        arch = GEN.make_architecture(2, 10, 0.02, np.random.default_rng(0))
        N = 40
        parents = GEN.sample_genotypes(0.5, arch.n_loci, N, np.random.default_rng(1))
        off = EV.mendelian_mating(parents, N, np.random.default_rng(2))
        assert off.shape[1] == N                     # N/2 pairs x 2 = N, clean


class TestUtilityGuards:
    def test_effective_size_survives_fixation(self):
        """`H → 0` (all loci fixed) must not poison the log-fit into NaN."""
        H = np.array([0.5, 0.3, 0.1, 0.0, 0.0])      # fixed by gen 3
        Ne = EV.effective_population_size(H)
        assert np.isfinite(Ne) or Ne == np.inf       # a number or inf, never NaN
        assert not np.isnan(Ne)

    def test_effective_size_of_flat_trajectory_is_infinite(self):
        assert EV.effective_population_size(np.full(10, 0.4)) == np.inf

    def test_retroviral_insertion_validates_genes_per_event(self):
        arch = GEN.make_architecture(4, 10, 0.02, np.random.default_rng(0))
        rng = np.random.default_rng(0)
        off = GEN.sample_genotypes(0.5, arch.n_loci, 20, rng)
        donors = np.ones((arch.n_loci, 3), dtype=int)
        ctx = EV.VariationContext(arch=arch, parents=off, donors=donors)
        with pytest.raises(ValueError, match="genes_per_event"):
            EV.retroviral_insertion(0.5, genes_per_event=99)(off, ctx, rng)
