"""Phase 1 gate #1: implicit-diff sensitivity vs finite differences, on the
mechanical engine.

Prints the numbers behind docs/DESIGN.md §3b:

  * ``equilibrate`` reaches a real fixed point (max|F| ~ 1e-14), where
    ``center_based.relax`` orbits forever at |grad| = 3.34;
  * the Hessian has exactly 3 zero modes (2 translations + 1 rotation);
  * implicit-diff matches finite differences to ~1e-9 on the gauge-invariant
    subspace, with dense-pinv and matrix-free CG agreeing to ~1e-11;
  * the discarded component is *entirely rotation* — the anholonomy that makes
    the Procrustes readout necessary rather than merely tidy.

Run:  python examples/demo_phase1_gate.py
"""
import jax

jax.config.update("jax_enable_x64", True)   # must precede any jax array work

import numpy as np
import jax.numpy as jnp

from jax_morpho.center_based import morse_energy, relax
from jax_morpho.evodevo import fixed_point as FP
from jax_morpho.evodevo import mechanical as M


def main():
    # -- the bug that motivates the engine ---------------------------------
    print("== center_based.relax: a period-2 limit cycle, not an equilibrium ==")
    n = 12
    rng = np.random.default_rng(0)
    p0 = jnp.asarray(rng.normal(0, 1.0, (n, 2)))
    al = jnp.ones(n)
    args = (1.0, 2.5, 1.0, 1.8)
    gn = lambda p: float(jnp.linalg.norm(jax.grad(morse_energy)(p, al, *args)))
    for steps in (200, 5_000, 100_000):
        print(f"   n_steps={steps:7d}   |grad| = {gn(relax(p0, al, *args, steps)):.6f}")
    x = relax(p0, al, *args, 100_000)
    print(f"   |p_k+1 - p_k| = {float(jnp.linalg.norm(relax(x, al, *args, 1) - x)):.3e}"
          f"   (it moves)")
    print(f"   |p_k+2 - p_k| = {float(jnp.linalg.norm(relax(x, al, *args, 2) - x)):.3e}"
          f"   (and comes right back)")

    # -- a real equilibrium on a cohesive tissue ---------------------------
    print("\n== evodevo.mechanical.equilibrate: a real fixed point ==")
    P = M.hex_blob(2)
    n = P.shape[0]
    rng = np.random.default_rng(1)
    pos0 = jnp.asarray(np.asarray(P) + rng.normal(0, 0.02, (n, 2)))
    alive = jnp.ones(n)
    theta = M.pack_theta(jnp.asarray(1.0 + 0.15 * rng.uniform(-1, 1, n)),
                         jnp.asarray(1.0 + 0.10 * rng.uniform(-1, 1, n)))
    xs, res, it, ok = M.equilibrate(pos0, alive, theta)
    print(f"   {n} cells, non-uniform theta ({theta.shape[0]} parameters)")
    print(f"   converged={bool(ok)}   max|F| = {float(res):.3e}   iters = {int(it)}")

    H = jax.hessian(lambda q: M.field_morse_energy(
        q.reshape(n, 2), alive, theta))(xs.ravel())
    w = np.linalg.eigvalsh(np.asarray(H))
    print(f"   Hessian spectrum:  {np.round(w[:6], 8)} ...  max {w[-1]:.1f}")
    print(f"   zero modes: {int((np.abs(w) < 1e-7).sum())}"
          f"  (2 translations + 1 rotation — the energy is rigid-invariant)")

    # -- gate #1 -----------------------------------------------------------
    print("\n== GATE #1: implicit-diff vs finite differences ==")
    develop = lambda th: M.equilibrate(pos0, alive, th)[0]
    J_fd = np.asarray(FP.finite_difference_sensitivity(develop, theta, eps=1e-6))
    J_pinv = np.asarray(FP.energy_sensitivity(M.field_morse_energy, xs, alive,
                                              theta, solver="pinv"))
    J_cg = np.asarray(FP.energy_sensitivity(M.field_morse_energy, xs, alive,
                                            theta, solver="cg"))

    Z = np.asarray(FP.rigid_modes(xs, alive))
    Proj = np.eye(Z.shape[0]) - Z @ Z.T
    rel = np.linalg.norm(Proj @ (J_pinv - J_fd)) / np.linalg.norm(Proj @ J_fd)
    print(f"   gauge-invariant subspace:  rel. diff = {rel:.3e}   <-- the gate")
    print(f"   dense pinv vs matrix-free CG:  rel. diff = "
          f"{np.linalg.norm(J_cg - J_pinv) / np.linalg.norm(J_pinv):.3e}")

    # -- the anholonomy ----------------------------------------------------
    print("\n== the discarded component is entirely rotation ==")
    relp = np.asarray(xs) - np.asarray(xs).mean(0)
    tx = np.stack([np.ones(n), np.zeros(n)], -1).ravel() / np.sqrt(n)
    ty = np.stack([np.zeros(n), np.ones(n)], -1).ravel() / np.sqrt(n)
    rot = np.stack([-relp[:, 1], relp[:, 0]], -1).ravel()
    rot /= np.linalg.norm(rot)
    for name, m in (("translation-x", tx), ("translation-y", ty), ("rotation", rot)):
        print(f"   {name:14s} component of J_fd:  {np.linalg.norm(m @ J_fd):.4e}")
    print(f"   |COM(x*) - COM(x0)| = "
          f"{float(jnp.linalg.norm(xs.mean(0) - pos0.mean(0))):.3e}"
          f"   (translation is exactly conserved)")
    print("\n   Zero net torque at every step, yet rotation accumulates: the modes"
          "\n   turn with the shape as it deforms — a geometric phase (the falling-cat"
          "\n   effect). The equilibrium FORM is a function of theta; its ORIENTATION"
          "\n   is a functional of the whole developmental path. Hence Procrustes.")


if __name__ == "__main__":
    main()
