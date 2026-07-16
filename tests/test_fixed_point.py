"""Phase 1: the implicit-diff sensitivity engine on the mechanical engine.

Gate #1 of the docs/DESIGN.md validation ladder: *the implicit-diff sensitivity
must equal the finite-difference Jacobian*. Finite differences know nothing
about the implicit machinery — they just re-run development — which is what
makes them an independent referee.

The gate is stated on the **gauge-invariant** part of the Jacobian, and
``TestGaugeIsRotationOnly`` is what earns that qualifier: it shows the discarded
part is purely a rotation and nothing else is being swept under the rug.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_morpho.center_based import morse_energy, relax
from jax_morpho.evodevo import fixed_point as FP
from jax_morpho.evodevo import mechanical as M

TOL = 1e-12


@pytest.fixture(scope="module")
def tissue():
    """A converged, cohesive 19-cell tissue with a non-uniform θ field."""
    P = M.hex_blob(2)
    n = P.shape[0]
    rng = np.random.default_rng(1)
    pos0 = jnp.asarray(np.asarray(P) + rng.normal(0, 0.02, (n, 2)))
    alive = jnp.ones(n)
    theta = M.pack_theta(jnp.asarray(1.0 + 0.15 * rng.uniform(-1, 1, n)),
                         jnp.asarray(1.0 + 0.10 * rng.uniform(-1, 1, n)))
    x, res, _, ok = M.equilibrate(pos0, alive, theta, tol=TOL)
    assert bool(ok), f"fixture failed to equilibrate: max|F| = {float(res):.3e}"
    return dict(pos0=pos0, alive=alive, theta=theta, x=x, n=n)


# ---------------------------------------------------------------------------
# The equilibrium itself
# ---------------------------------------------------------------------------

class TestEquilibrium:
    def test_equilibrate_reaches_machine_precision(self, tissue):
        x, alive, theta = tissue["x"], tissue["alive"], tissue["theta"]
        res = jnp.abs(M.force_residual(x, alive, theta)).max()
        assert float(res) < TOL

    def test_relax_never_converges_period_2_limit_cycle(self):
        """Regression: pin the bug that motivates ``equilibrate``.

        ``center_based.relax`` does not reach a fixed point. Its clipped fixed
        step turns the instability of the stiffest mode into a *stable* period-2
        orbit, so it looks converged while ``|∇E|`` never falls. If this test
        ever starts failing, ``relax`` was fixed and DESIGN.md §1 needs revisiting.
        """
        n = 12
        rng = np.random.default_rng(0)
        pos = jnp.asarray(rng.normal(0, 1.0, (n, 2)))
        alive = jnp.ones(n)
        args = (1.0, 2.5, 1.0, 1.8)
        gnorm = lambda p: float(jnp.linalg.norm(
            jax.grad(morse_energy)(p, alive, *args)))

        # 500x more steps buys nothing: it is orbiting, not converging.
        short, long = relax(pos, alive, *args, 200), relax(pos, alive, *args, 100_000)
        assert gnorm(short) > 1.0
        assert abs(gnorm(long) - gnorm(short)) < 1e-3

        # ...and the orbit is exactly period 2.
        p1 = relax(long, alive, *args, 1)
        p2 = relax(long, alive, *args, 2)
        assert float(jnp.linalg.norm(p1 - long)) > 1e-3      # it does move
        assert float(jnp.linalg.norm(p2 - long)) < 1e-12     # ...and comes back


# ---------------------------------------------------------------------------
# Gauge structure
# ---------------------------------------------------------------------------

class TestCGNewton:
    """The matrix-free Newton direction — the only solver that survives scale.

    The dense path's cost is O(N³) *memory*: ``jax.hessian`` differentiates
    through ``field_morse_energy``'s O(N²) pair matrix and builds a ``(2N, N, N)``
    intermediate (32 GB at N=1261, where the Hessian itself is 51 MB). CG needs
    only Hessian-vector products.
    """

    def test_cg_and_pinv_reach_the_same_form(self, tissue):
        """Not the same *coordinates* — the same *organism*.

        The two solvers take different paths to the same basin, so they differ by
        a rigid motion: the developmental anholonomy again (§ TestGaugeIsRotationOnly).
        Measured: the difference is ~1e-5 and lives **entirely** in the rigid
        modes, with the physical component a million times smaller — and the
        Procrustes phenotype is identical to machine precision. Swapping the
        linear solver inside Newton moves the orientation and nothing else.
        """
        pos0, alive, theta = (tissue[k] for k in ("pos0", "alive", "theta"))
        xp, _, _, ok_p = M.equilibrate(pos0, alive, theta, tol=TOL,
                                       newton_solver="pinv")
        xc, _, _, ok_c = M.equilibrate(pos0, alive, theta, tol=TOL,
                                       newton_solver="cg")
        assert bool(ok_p) and bool(ok_c)

        diff = np.asarray(xc - xp).ravel()
        Z = np.asarray(FP.rigid_modes(xp, alive))
        rigid = Z @ (Z.T @ diff)
        physical = diff - rigid
        assert np.linalg.norm(diff) > 1e-9                    # they *do* differ
        assert np.linalg.norm(physical) < 1e-9                # ...but not in form
        assert np.linalg.norm(physical) < 1e-3 * np.linalg.norm(rigid)

    def test_phenotype_is_solver_independent(self, tissue):
        """The property that makes the readout worth its cost: change the
        numerics, and the science does not move."""
        from jax_morpho.evodevo import phenotype as PH

        pos0, alive, theta, n = (tissue[k] for k in ("pos0", "alive", "theta", "n"))
        idx = jnp.arange(n)
        xp = M.equilibrate(pos0, alive, theta, tol=TOL, newton_solver="pinv")[0]
        xc = M.equilibrate(pos0, alive, theta, tol=TOL, newton_solver="cg")[0]
        ref = PH.make_reference(xp, idx)
        zp = PH.procrustes_shape(xp, idx, ref)
        zc = PH.procrustes_shape(xc, idx, ref)
        assert float(jnp.linalg.norm(zc - zp)) < 1e-12

    def test_cg_handles_dead_cells(self):
        """``pinv`` absorbed the padded cells' null directions for free; CG has
        to be told about them (the operator acts as the identity there)."""
        P = M.hex_blob(1)
        n_live = P.shape[0]
        pos = jnp.concatenate([P, jnp.array([[9.0, 9.0], [11.0, 9.0]])])
        alive = jnp.concatenate([jnp.ones(n_live), jnp.zeros(2)])
        x, res, _, ok = M.equilibrate(pos, alive, M.uniform_theta(n_live + 2),
                                      tol=TOL, newton_solver="cg")
        assert bool(ok), f"CG failed with dead cells: max|F| = {float(res):.3e}"
        assert float(jnp.abs(x[n_live:] - pos[n_live:]).max()) < 1e-14

    def test_unknown_newton_solver_rejected(self, tissue):
        with pytest.raises(ValueError, match="unknown newton_solver"):
            M.equilibrate(tissue["pos0"], tissue["alive"], tissue["theta"],
                          newton_solver="dense-ish")


class TestGauge:
    def test_exactly_three_rigid_zero_modes(self, tissue):
        x, alive, theta, n = (tissue[k] for k in ("x", "alive", "theta", "n"))
        H = jax.hessian(lambda q: M.field_morse_energy(
            q.reshape(n, 2), alive, theta))(x.ravel())
        w = np.linalg.eigvalsh(np.asarray(H))
        # 2 translations + 1 rotation, and nothing else: a cohesive tissue has
        # no floating fragments contributing extra zero modes.
        assert int((np.abs(w) < 1e-7).sum()) == 3
        assert w[3] > 1e-2        # a real gap; the rest of the spectrum is stiff

    def test_rigid_modes_are_the_zero_modes(self, tissue):
        x, alive, theta, n = (tissue[k] for k in ("x", "alive", "theta", "n"))
        H = jax.hessian(lambda q: M.field_morse_energy(
            q.reshape(n, 2), alive, theta))(x.ravel())
        Z = FP.rigid_modes(x, alive)
        assert np.abs(np.asarray(H @ Z)).max() < 1e-8       # H annihilates them
        assert np.abs(np.asarray(Z.T @ Z) - np.eye(3)).max() < 1e-10   # orthonormal


# ---------------------------------------------------------------------------
# Gate #1
# ---------------------------------------------------------------------------

class TestGate1ImplicitVsFiniteDifference:
    def test_implicit_matches_finite_differences(self, tissue):
        """The gate. Implicit-diff ⟺ finite differences, on the physical
        (gauge-invariant) subspace."""
        pos0, alive, theta, x = (tissue[k] for k in ("pos0", "alive", "theta", "x"))
        develop = lambda th: M.equilibrate(pos0, alive, th, tol=TOL)[0]
        J_fd = np.asarray(FP.finite_difference_sensitivity(develop, theta, eps=1e-6))
        J_im = np.asarray(FP.energy_sensitivity(
            M.field_morse_energy, x, alive, theta, solver="pinv"))

        Z = np.asarray(FP.rigid_modes(x, alive))
        P = np.eye(Z.shape[0]) - Z @ Z.T
        rel = np.linalg.norm(P @ (J_im - J_fd)) / np.linalg.norm(P @ J_fd)
        assert rel < 1e-6, f"gate #1 failed: relative difference {rel:.3e}"

    def test_sensitivity_is_nontrivial(self, tissue):
        """Guard against the gate passing on an accidentally-zero Jacobian."""
        J = FP.energy_sensitivity(M.field_morse_energy, tissue["x"],
                                  tissue["alive"], tissue["theta"])
        assert float(jnp.abs(J).max()) > 1e-2

    def test_implicit_jacobian_has_no_rigid_component(self, tissue):
        """The implicit sensitivity lives entirely in the physical subspace —
        by construction, since the pseudo-inverse takes the minimum-norm step."""
        x, alive, theta = (tissue[k] for k in ("x", "alive", "theta"))
        J = FP.energy_sensitivity(M.field_morse_energy, x, alive, theta)
        Z = FP.rigid_modes(x, alive)
        assert float(jnp.linalg.norm(Z.T @ J)) < 1e-8


class TestGaugeIsRotationOnly:
    def test_discrepancy_is_purely_rotational(self, tissue):
        """Why gate #1 is allowed to project.

        Relaxation conserves the centre of mass *exactly* (pairwise central
        forces sum to zero), so no translation enters. It does **not** conserve
        orientation: each step carries zero net torque, yet the tissue
        accumulates a net rotation as its shape changes along the path — a
        geometric phase, the falling-cat effect. So the equilibrium *form* is a
        function of θ while its *orientation* is a functional of the whole
        developmental trajectory, which no fixed-point method can or should
        reproduce.

        This is why the phenotype readout is Procrustes-aligned landmarks
        (DESIGN.md §2C): the alignment is not tidying-up, it is what makes the
        phenotype a well-defined function of the genotype at all.
        """
        pos0, alive, theta, n = (tissue[k] for k in ("pos0", "alive", "theta", "n"))
        x = tissue["x"]
        develop = lambda th: M.equilibrate(pos0, alive, th, tol=TOL)[0]
        J_fd = np.asarray(FP.finite_difference_sensitivity(develop, theta, eps=1e-6))

        rel = np.asarray(x) - np.asarray(x).mean(0)
        tx = np.stack([np.ones(n), np.zeros(n)], -1).ravel() / np.sqrt(n)
        ty = np.stack([np.zeros(n), np.ones(n)], -1).ravel() / np.sqrt(n)
        rot = np.stack([-rel[:, 1], rel[:, 0]], -1).ravel()
        rot /= np.linalg.norm(rot)

        # Translation: absent from the response (COM is conserved along the path).
        assert np.linalg.norm(tx @ J_fd) < 1e-5
        assert np.linalg.norm(ty @ J_fd) < 1e-5
        # Rotation: present, and large — the anholonomy.
        assert np.linalg.norm(rot @ J_fd) > 1e-1

    def test_relaxation_conserves_centre_of_mass(self, tissue):
        pos0, x = tissue["pos0"], tissue["x"]
        assert float(jnp.linalg.norm(x.mean(0) - pos0.mean(0))) < 1e-12


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

class TestSolvers:
    def test_matrix_free_cg_matches_dense_pinv(self, tissue):
        """The path that scales must agree with the path that is obviously
        right: CG never forms the Hessian, only Hessian-vector products."""
        x, alive, theta = (tissue[k] for k in ("x", "alive", "theta"))
        Jp = FP.energy_sensitivity(M.field_morse_energy, x, alive, theta,
                                   solver="pinv")
        Jc = FP.energy_sensitivity(M.field_morse_energy, x, alive, theta,
                                   solver="cg")
        rel = float(jnp.linalg.norm(Jc - Jp) / jnp.linalg.norm(Jp))
        assert rel < 1e-6

    def test_cg_requires_a_null_basis(self, tissue):
        F = lambda x, th: M.force_residual(x, tissue["alive"], th)
        with pytest.raises(ValueError, match="null_basis"):
            FP.fixed_point_sensitivity(F, tissue["x"], tissue["theta"],
                                       solver="cg")

    def test_unknown_solver_rejected(self, tissue):
        F = lambda x, th: M.force_residual(x, tissue["alive"], th)
        with pytest.raises(ValueError, match="unknown solver"):
            FP.fixed_point_sensitivity(F, tissue["x"], tissue["theta"],
                                       solver="newton-ish")


# ---------------------------------------------------------------------------
# The energy itself
# ---------------------------------------------------------------------------

class TestFieldEnergy:
    def test_uniform_field_reproduces_center_based_forces(self):
        """A uniform θ field must reproduce the *forces* of the global-parameter
        Morse potential in ``center_based``, for pairs inside ``r_on``.

        Forces, not energies: ``center_based`` subtracts a constant from every
        interacting pair to make its truncation continuous, so the two energies
        differ by that offset (here 3 pairs x 0.252). A constant has no
        gradient, so the mechanics — which is all the equilibrium sees — is
        identical.

        Uses a scalene triangle with every side inside ``r_on = 1.44``, so the
        switch is identically 1 and the comparison is exact. (A hex lattice
        cannot serve: its second-neighbour distance √3 = 1.732 lands inside the
        switching shell, which is exactly where the two are *meant* to differ.)
        """
        P = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.35, 0.95]])
        n = 3
        alive = jnp.ones(n)
        D, r_eq, a, r_max = 1.0, 1.0, 2.5, 1.8

        r = jnp.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        off = np.asarray(r)[~np.eye(n, dtype=bool)]
        assert (off < M.R_ON_FRAC * r_max).all(), "pairs must clear the switch"

        f_field = jax.grad(M.field_morse_energy)(
            P, alive, M.uniform_theta(n, D, r_eq), a, r_max)
        f_center = jax.grad(morse_energy)(P, alive, D, a, r_eq, r_max)
        # ~1e-8, not machine zero: center_based softens its sqrt with +1e-9
        # where this module uses +1e-12, which shifts r — and so the force —
        # at exactly that order.
        assert float(jnp.abs(f_field - f_center).max()) < 1e-7
        assert float(jnp.abs(f_center).max()) > 1e-3      # non-trivial forces

        # ...and the energies differ by exactly the per-pair shift.
        shift = D * ((1.0 - jnp.exp(-a * (r_max - r_eq))) ** 2 - 1.0)
        e_field = M.field_morse_energy(P, alive, M.uniform_theta(n, D, r_eq),
                                       a, r_max)
        e_center = morse_energy(P, alive, D, a, r_eq, r_max)
        assert abs(float(e_field - e_center) - 3 * float(shift)) < 1e-9

    def test_switch_takes_the_potential_smoothly_to_zero(self):
        """Beyond ``r_max`` the pair energy and, unlike ``center_based``, its
        *force* both vanish — the C² cutoff that makes the Hessian well posed."""
        pair = lambda d: M.field_morse_energy(
            jnp.array([[0.0, 0.0], [d, 0.0]]), jnp.ones(2), M.uniform_theta(2))
        assert abs(float(pair(1.9))) < 1e-12                  # beyond cutoff
        force_at_cutoff = float(jax.grad(pair)(1.8 - 1e-6))
        assert abs(force_at_cutoff) < 1e-6                    # and no force jump

    def test_energy_is_rigid_invariant(self, tissue):
        """The symmetry that creates the zero modes in the first place."""
        x, alive, theta = (tissue[k] for k in ("x", "alive", "theta"))
        e0 = M.field_morse_energy(x, alive, theta)
        shifted = x + jnp.array([3.7, -1.2])
        c, s = jnp.cos(0.9), jnp.sin(0.9)
        rotated = x @ jnp.array([[c, -s], [s, c]])
        assert abs(float(M.field_morse_energy(shifted, alive, theta) - e0)) < 1e-9
        assert abs(float(M.field_morse_energy(rotated, alive, theta) - e0)) < 1e-9

    def test_padded_dead_cells_do_not_move_or_contribute(self):
        """Padded slots must be inert: fixed-size arrays are how this stays
        jit- and GPU-friendly, so dead cells must not perturb the equilibrium."""
        P = M.hex_blob(1)
        n_live = P.shape[0]
        pos = jnp.concatenate([P, jnp.array([[9.0, 9.0], [11.0, 9.0]])])
        alive = jnp.concatenate([jnp.ones(n_live), jnp.zeros(2)])
        theta = M.uniform_theta(n_live + 2)

        x, res, _, ok = M.equilibrate(pos, alive, theta, tol=TOL)
        assert bool(ok)
        # dead cells stayed exactly put
        assert float(jnp.abs(x[n_live:] - pos[n_live:]).max()) < 1e-14
        # and the live equilibrium ignores them entirely
        x_only, _, _, ok2 = M.equilibrate(P, jnp.ones(n_live),
                                          M.uniform_theta(n_live), tol=TOL)
        assert bool(ok2)
        assert float(jnp.abs(x[:n_live] - x_only).max()) < 1e-8
