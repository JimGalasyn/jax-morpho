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
    print("        p    angle_G   angle_P   noise floor   snr    h^2   verdict")
    opt = RS.reference_optimum(org, arch)
    for p in (0.5, 0.25, 0.125, 0.0625, 0.03125):
        r = RS.simulate_response(p, org, arch, optimum=opt, n_ind=700,
                                 n_replays=8, seed=0)
        ok = r["snr"] > 3.0
        verdict = ("G at the floor" if ok and r["angle_G"] < 2.5 * r["noise_angle"]
                   else "resolved" if ok else "NOISE — response below the floor")
        print(f"   {r['p']:7.5f}  {r['angle_G']:6.2f}°  {r['angle_P']:7.2f}°"
              f"   {r['noise_angle']:8.2f}°  {r['snr']:5.1f}  {r['heritability']:5.2f}"
              f"   {verdict}")

    print("\n   G's error sits AT the noise floor -> consistent with exact.")
    print("   P's is an order of magnitude above it -> a real systematic error.")
    print("   Below p~0.125 the response falls under the noise floor: the")
    print("   response shrinks with 2pq while the noise (set by the environment-")
    print("   dominated phenotypic sd) does not. M-U buy that tail with 5000 x 50")
    print("   replays ~ 5e5 developments/point; we spend ~1e4. And it CANNOT be")
    print("   bought by raising sigma_gamma: that lifts snr but breaks the linear")
    print("   regime G lives in (measured: 0.02->0.08 sends angle_G 3.3deg -> 18.8deg).")

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
