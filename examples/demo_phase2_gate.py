"""Phase 2: the GRN genome map, the Procrustes phenotype, and gate #2.

Prints the numbers behind docs/DESIGN.md §3c:

  * the composed genome→shape Jacobian matches finite differences **raw**, with
    no gauge projection — the payoff of Phase 1's anholonomy finding, and the
    reason the Procrustes readout is load-bearing rather than cosmetic;
  * gate #2: G = J M Jᵀ reproduces the covariance of a developed population, and
    the discrepancy shrinks with the genetic spread;
  * development is multistable — past a threshold, individuals jump to a
    different packing and the phenotype is discontinuous in the genome. G is a
    within-basin object.

Run:  python examples/demo_phase2_gate.py
"""
import jax

jax.config.update("jax_enable_x64", True)   # must precede any jax array work

import numpy as np
import jax.numpy as jnp

from jax_morpho.evodevo import fixed_point as FP
from jax_morpho.evodevo import genome_map as GM
from jax_morpho.evodevo import mechanical as M
from jax_morpho.evodevo import phenotype as PH
from jax_morpho.evodevo import pipeline as PL
from jax_morpho.evodevo import quantgen as QG

N_GENES, N_SAMPLES = 4, 400


def main():
    grn = GM.init_grn(jax.random.key(0), N_GENES, hidden=16, scale=1.5)
    a0 = jnp.zeros(N_GENES)
    org = PL.make_organism(grn, a0, n_rings=2)
    n, k = org.pos0.shape[0], org.idx.shape[0]

    print("== the pipeline: genome -> theta field -> equilibrium -> shape ==")
    D, r_eq = M.unpack_theta(PL.theta_of(a0, org))
    print(f"   {N_GENES} genes -> GRN -> per-cell field over {n} cells")
    print(f"   D    in [{D.min():.3f}, {D.max():.3f}]   (bounds {GM.D_LO}, {GM.D_HI})")
    print(f"   r_eq in [{r_eq.min():.3f}, {r_eq.max():.3f}]   (bounds "
          f"{GM.R_EQ_LO}, {GM.R_EQ_HI})")
    print(f"   -> {k} landmarks -> Procrustes shape z, dim {2*k} "
          f"(shape space {PH.shape_dim(k)} = 2k-4)")

    # -- the Phase 1 payoff ------------------------------------------------
    print("\n== Phase 1 payoff: the composed chain needs NO gauge projection ==")
    J = PL.phenotype_jacobian(a0, org)
    J_fd = FP.finite_difference_sensitivity(lambda a: PL.phenotype(a, org),
                                            eps=1e-6, theta=a0)
    rel = float(jnp.linalg.norm(J - J_fd) / jnp.linalg.norm(J_fd))
    print(f"   dz/da vs finite differences, RAW:  {rel:.3e}")
    x = PL.develop(a0, org)
    Z = FP.rigid_modes(x, org.alive)
    print(f"   readout annihilates the rigid modes: |dz/dx* @ Z| = "
          f"{float(jnp.abs(PH.shape_jacobian(x, org.idx, org.ref) @ Z).max()):.2e}")
    print("   (Phase 1's RAW dx*/dtheta vs FD failed at ~0.7 — the anholonomic")
    print("    rotation. Procrustes quotients it out, so the chain agrees raw.)")

    # -- gate #2 -----------------------------------------------------------
    print(f"\n== GATE #2: G = J M J^T  vs  empirical Cov(z)   (n={N_SAMPLES}, "
          f"common random numbers) ==")
    xi = jax.random.normal(jax.random.key(7), (N_SAMPLES, N_GENES))
    print("   sigma        rel diff")
    for sigma in (1e-2, 5e-3, 2.5e-3, 1.25e-3):
        A = a0 + sigma * xi
        Zp = PL.phenotype_population(A, org)
        rel = QG.relative_difference(QG.empirical_covariance(Zp),
                                     QG.build_G(J, QG.empirical_covariance(A)))
        print(f"   {sigma:<10.5f}   {rel:.4e}")
    print("   The discrepancy is controlled by sigma, as a local claim must be.")
    print("   (Its fitted ORDER is not a stable statistic here: two error terms")
    print("    compete — an O(sigma^2) truncation and an O(sigma) finite-sample")
    print("    term — so the fit swings between ~0.8 and ~2.0 with sample size.)")

    G = QG.build_G(J, jnp.eye(N_GENES))
    print(f"\n   rank(G) = {np.linalg.matrix_rank(np.asarray(G), tol=1e-12)}"
          f"  = n_genes ({N_GENES}): development cannot express more independent"
          f"\n   directions of variation than the genome supplies.")

    # -- multistability ----------------------------------------------------
    print("\n== development is multistable: G is a within-basin object ==")
    x0 = PL.develop(a0, org)
    for sigma in (1.25e-3, 0.05):
        X, _, _ = PL.develop_population(a0 + sigma * xi[:120], org)
        dx = np.array([float(jnp.linalg.norm(xx - x0)) for xx in X])
        jumped = dx > 0.2
        print(f"   sigma={sigma:<8.5f}  jumped to another packing: "
              f"{jumped.sum():3d}/120   |x*-x0| median "
              f"{np.median(dx):.4f}" +
              (f", jumpers {np.median(dx[jumped]):.4f}" if jumped.any() else ""))
    print("\n   At sigma=0.05 the form displacement is bimodal (~0.02 vs ~0.43):")
    print("   a neighbour exchange, so the phenotype is DISCONTINUOUS in the")
    print("   genome and no local Jacobian can describe the crossing. Fittingly,")
    print("   Milocco & Uller's own reference model is a bistable toggle switch:")
    print("   their development is multistable by construction, ours by consequence.")


if __name__ == "__main__":
    main()
