"""Phase 4 gate #4: the evolution loop over time.

Prints the numbers behind docs/DESIGN.md §3e:

  * gate #4a — neutral drift decays heterozygosity geometrically, and the
    effective population size is ~2N (equal family size), not N;
  * gate #4b — under selection the mean tracks the optimum and plateaus as
    genetic variance is exhausted, and the per-generation breeder's prediction
    Gβ tracks the realised response WHILE variance is healthy, then fails once
    β = P⁻¹s destabilises;
  * the §5c seams — point mutation (within-basin) vs retroviral insertion
    (whole-gene, from a donor lineage: the basin-jump).

Run:  python examples/demo_phase4_gate.py
"""
import jax

jax.config.update("jax_enable_x64", True)   # must precede any jax array work

import numpy as np
import jax.numpy as jnp

from jax_morpho.evodevo import evolution as EV
from jax_morpho.evodevo import genetics as GEN
from jax_morpho.evodevo import genome_map as GM
from jax_morpho.evodevo import pipeline as PL
from jax_morpho.evodevo import quantgen as QG
from jax_morpho.evodevo import response as RS

N_GENES, N_ENV = 4, 2


def main():
    # -- gate #4a: drift and Ne ---------------------------------------------
    print("== GATE #4a: neutral drift -> Ne ~ 2N (equal family size), not N ==")
    arch1 = GEN.make_architecture(1, 60, 0.02, np.random.default_rng(0))
    print("   N     Ne (fitted)   Ne/N   (Wright-Fisher would say Ne/N=1)")
    for N in (20, 40, 80):
        H = np.mean([
            [h["heterozygosity"] for h in EV.evolve(
                GEN.sample_genotypes(0.5, arch1.n_loci, N, np.random.default_rng(s)),
                None, arch1, n_generations=40,
                selection=EV.neutral_selection(1.0), develop=False, seed=s)[1]]
            for s in range(30)], axis=0)
        Ne = EV.effective_population_size(H)
        print(f"   {N:<4d}    {Ne:6.1f}       {Ne / N:.2f}")
    print("   Every pair makes exactly 2 offspring -> zero family-size variance")
    print("   -> Ne ~ 2N-1. Drift is HALF as strong as naive Wright-Fisher.")

    # -- gate #4b: selection, the limit, and where the prediction fails ------
    print("\n== GATE #4b: does the one-generation prediction compound? ==")
    grn = GM.init_grn(jax.random.key(0), N_GENES + N_ENV, hidden=16, scale=1.5)
    org = PL.make_organism(grn, jnp.zeros(N_GENES + N_ENV), n_rings=2,
                           landmark_stride=5)
    arch = GEN.make_architecture(N_GENES, 5, 0.04, np.random.default_rng(0))
    opt = RS.reference_optimum(org, arch, sigma_env=0.005, optimum_scale=4.0)

    sc = GEN.sample_genotypes(0.5, arch.n_loci, 200, np.random.default_rng(0))
    _, hist = EV.evolve(sc, org, arch, n_generations=12,
                        selection=EV.truncation_toward(opt, 0.2),
                        sigma_env=0.005, measure_response=True, seed=0)
    g = [h for h in hist if h.get("z_mean") is not None]
    z0 = g[0]["z_mean"]
    u = (opt - z0) / np.linalg.norm(opt - z0)
    print("   (single seed -> per-gen angle is noisy; the 8-seed binned medians")
    print("    are 21° at het≈0.48, 40° at 0.40, 78° at 0.30 -- test_evolution.py)")
    print("   gen   toward-opt   het    per-gen angle(Gβ, realised)")
    for i in range(len(g) - 1):
        prog = (g[i]["z_mean"] - z0) @ u
        obs = g[i + 1]["z_mean"] - g[i]["z_mean"]
        pred = g[i]["dz_pred"]
        ang = (np.degrees(np.arccos(np.clip(
            obs @ pred / (np.linalg.norm(obs) * np.linalg.norm(pred) + 1e-30),
            -1, 1))) if np.linalg.norm(obs) > 1e-9 else float("nan"))
        print(f"   {i:>3d}   {prog:>+.3e}   {g[i]['heterozygosity']:.3f}   {ang:5.0f}°")
    print("\n   The mean advances on the optimum and PLATEAUS as heterozygosity")
    print("   collapses (a heritability-limited selection limit). Gβ predicts the")
    print("   response while variance is fresh (~20°) and DEGRADES as it runs out")
    print("   -- not a defect, the honest boundary: β = P⁻¹s destabilises as P")
    print("   approaches singular. The loop recomputes G each generation because a")
    print("   G frozen at gen 0 describes a population that no longer exists.")

    # -- §5c seams ----------------------------------------------------------
    print("\n== the §5c variation seams: gradualism vs punctuation ==")
    rng = np.random.default_rng(0)
    off = GEN.sample_genotypes(0.5, arch.n_loci, 100, rng)
    donors = np.ones((arch.n_loci, 5), dtype=int)         # a distinct donor lineage
    ctx = EV.VariationContext(arch=arch, parents=off, donors=donors)

    mutated = EV.point_mutation(0.05)(off, ctx, rng)
    print(f"   point mutation (rate 0.05): {100 * (mutated != off).mean():.1f}% of "
          f"loci changed -- a small, within-basin perturbation")

    infected = EV.retroviral_insertion(1.0, genes_per_event=1)(off, ctx, rng)
    whole_genes = np.mean([
        sum(bool(np.all(infected[arch.gene_of_locus == gene, i] == 1))
            for gene in range(arch.n_genes))
        for i in range(infected.shape[1])])
    print(f"   retroviral insertion: each offspring got {whole_genes:.1f} WHOLE gene(s)")
    print(f"   overwritten from the donor -- a coordinated multi-locus jump, the")
    print(f"   basin-crossing macromutation (DESIGN.md §5c). Needs a donor pool:")
    try:
        EV.retroviral_insertion(1.0)(off, EV.VariationContext(arch, off, None), rng)
    except ValueError as e:
        print(f"     without one -> ValueError: {str(e)[:60]}...")


if __name__ == "__main__":
    main()
