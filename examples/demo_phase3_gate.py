"""Phase 3 gate #3: the Fig-3C pattern, on *our* development.

Prints the numbers behind docs/DESIGN.md §3d:

  * the tangent shape space, which is what makes `β = P⁻¹s` well posed at all;
  * gate #3: G predicts the response to within the measurement's own noise
    floor, while P's error is an order of magnitude above it;
  * the full MAF sweep — **including the points where the measurement dies**,
    which is the honest half of the picture;
  * the reverse-mode path: `Δz̄ = J M Jᵀβ` in two solves, no G, no J.

Run:  python examples/demo_phase3_gate.py
"""
import jax

jax.config.update("jax_enable_x64", True)   # must precede any jax array work

import numpy as np
import jax.numpy as jnp

from jax_morpho.evodevo import genetics as GEN
from jax_morpho.evodevo import genome_map as GM
from jax_morpho.evodevo import phenotype as PH
from jax_morpho.evodevo import pipeline as PL
from jax_morpho.evodevo import quantgen as QG
from jax_morpho.evodevo import response as RS

N_GENES, N_ENV = 4, 2


def main():
    grn = GM.init_grn(jax.random.key(0), N_GENES + N_ENV, hidden=16, scale=1.5)
    org = PL.make_organism(grn, jnp.zeros(N_GENES + N_ENV), n_rings=2,
                           landmark_stride=5)
    arch = GEN.make_architecture(N_GENES, loci_per_gene=5, sigma_gamma=0.02,
                                 rng=np.random.default_rng(0))
    k = org.idx.shape[0]

    print("== the loop: loci -> genome -> theta field -> equilibrium -> shape -> G ==")
    print(f"   {arch.n_loci} diploid loci -> {N_GENES} genes (+{N_ENV} non-heritable"
          f" environmental inputs)")
    print(f"   -> GRN -> per-cell field -> {org.pos0.shape[0]}-cell tissue")
    print(f"   -> {k} landmarks -> shape z ({2*k}D), tangent space {PH.shape_dim(k)}D")

    # -- why the tangent space is not optional -----------------------------
    print("\n== P is singular in ambient shape coordinates — beta = P^-1 s needs "
          "the tangent space ==")
    rng = np.random.default_rng(0)
    A = np.concatenate([rng.normal(0, 0.01, (300, N_GENES)),
                        rng.normal(0, 0.02, (300, N_ENV))], 1)
    Z = PL.phenotype_population(jnp.asarray(A), org)
    zt = np.asarray(PH.tangent_coords(Z, org.ref))
    amb = np.asarray(QG.empirical_covariance(Z))
    tan = np.asarray(QG.empirical_covariance(jnp.asarray(zt)))
    print(f"   rank(P) ambient  ({2*k}x{2*k}) = "
          f"{np.linalg.matrix_rank(amb, tol=1e-14)}   <- singular by construction")
    print(f"   rank(P) tangent  ({tan.shape[0]}x{tan.shape[0]}) = "
          f"{np.linalg.matrix_rank(tan, tol=1e-14)}   <- invertible")

    # -- gate #3 -----------------------------------------------------------
    print("\n== GATE #3: does a development-derived G predict the response? ==")
    print("   genes 0,1 swept; genes 2,3 held at p=0.5; optimum FIXED across the sweep")
    print("   'noise floor' = arcsin(1/snr): the best ANY prediction could show\n")
    print("   THREE SEEDS shown, deliberately: a single seed can look like a clean")
    print("   monotone Fig-3C and it is a lucky draw. Read the spread, not a row.\n")
    opt = RS.reference_optimum(org, arch)
    ps = (0.5, 0.25, 0.125, 0.0625, 0.03125)
    for seed in (0, 1, 2):
        rs = [RS.simulate_response(p, org, arch, optimum=opt, n_ind=700,
                                   n_replays=8, seed=seed) for p in ps]
        mono = all(rs[i]["angle_P"] < rs[i + 1]["angle_P"] for i in range(len(rs) - 1))
        print(f"   seed={seed}   p:      " + "".join(f"{p:>9.4f}" for p in ps))
        print(f"            angle_G:" + "".join(f"{r['angle_G']:>8.1f}°" for r in rs))
        print(f"            floor:  " + "".join(f"{r['noise_angle']:>8.1f}°" for r in rs))
        print(f"            angle_P:" + "".join(f"{r['angle_P']:>8.1f}°" for r in rs)
              + f"   monotone={mono}")
        print(f"            snr:    " + "".join(f"{r['snr']:>9.1f}" for r in rs))
        # read the library's own verdict rather than re-deriving the threshold
        print(f"            valid?  " + "".join(
            f"{'  yes' if r['resolved'] else ' NOISE':>9}" for r in rs) + "\n")

    print("   READ THIS AS: at p >= 0.25, where snr is comfortable, G's error sits")
    print("   AT the noise floor -> consistent with EXACT, while P is 20-114deg off")
    print("   -> a real systematic error. That contrast is robust across seeds and")
    print("   is what gate #3 asserts.")
    print()
    print("   NOT claimed: the monotone growth of angle_P as p -> 0. Seed 0 shows it")
    print("   beautifully; seeds 1 and 2 do not. It is not a stable feature at these")
    print("   sample sizes -- the response shrinks with 2pq while the noise floor,")
    print("   set by the environment-dominated phenotypic sd, does not, so snr dies")
    print("   and the low-p angles are noise (watch the floor climb past 20deg).")
    print("   M-U buy that tail with 5000 x 50 replays ~ 5e5 developments/point;")
    print("   we spend ~1e4. And it CANNOT be bought by raising sigma_gamma: that")
    print("   lifts snr but breaks the linear regime G lives in (measured:")
    print("   0.02 -> 0.08 sends angle_G 3.3deg -> 18.8deg). Gate #2's small-")
    print("   perturbation constraint and gate #3's snr pull against each other.")

    # -- the reverse-mode path ---------------------------------------------
    print("\n== the reverse-mode path: dz = J M J^T beta, no G, no J ==")
    a = jnp.zeros(N_GENES + N_ENV)
    rng = np.random.default_rng(3)
    M = jnp.asarray(np.diag(rng.uniform(0.5, 2.0, N_GENES + N_ENV)))
    beta = jnp.asarray(rng.normal(0, 1, 2 * k))
    got = PL.lande_response_vjp(a, org, beta, M)
    J = PL.phenotype_jacobian(a, org)
    want = J @ (M @ (J.T @ beta))
    print(f"   two solves vs explicitly forming J: rel. diff = "
          f"{float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want)):.3e}")
    print(f"   forming J costs {a.shape[0]} solves (one per input); this costs 2,")
    print("   independent of gene AND trait count.")
    print("\n   It cannot be jax.vjp through the map: equilibrate is a lax.while_loop")
    print("   (reverse-mode undefined), and unrolling it would differentiate the")
    print("   relaxation PATH rather than the fixed point — Phase 1's finding. The")
    print("   implicit transpose is what makes a reverse-mode path exist here.")


if __name__ == "__main__":
    main()
