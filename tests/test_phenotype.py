"""Phase 2, layers A and C: the GRN genome map and the Procrustes phenotype.

The headline test here is ``TestGaugeInvarianceOfTheChain``, which closes the
Phase 1 finding: the raw ∂x*/∂θ disagreed with finite differences by exactly one
rotational mode (the developmental anholonomy), so gate #1 had to be stated on
the gauge-invariant subspace. Once the Procrustes readout is composed on top,
the full ∂z/∂a must match finite differences **raw, with no projection** — which
is the whole reason DESIGN.md §2C chose that readout.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_morpho.evodevo import fixed_point as FP
from jax_morpho.evodevo import genome_map as GM
from jax_morpho.evodevo import mechanical as M
from jax_morpho.evodevo import phenotype as PH
from jax_morpho.evodevo import pipeline as PL

N_GENES = 4


@pytest.fixture(scope="module")
def org():
    grn = GM.init_grn(jax.random.key(0), N_GENES, hidden=16, scale=1.5)
    return PL.make_organism(grn, jnp.zeros(N_GENES), n_rings=2)


@pytest.fixture(scope="module")
def a0():
    return jnp.zeros(N_GENES)


# ---------------------------------------------------------------------------
# Layer A — the genome map
# ---------------------------------------------------------------------------

class TestGenomeMap:
    def test_produces_a_bounded_non_uniform_field(self, org, a0):
        theta = PL.theta_of(a0, org)
        D, r_eq = M.unpack_theta(theta)
        # a field, not a constant: the genome addresses cells differently
        assert float(D.std()) > 1e-3
        assert float(r_eq.std()) > 1e-3
        # and strictly inside the bounds, for every cell
        assert bool((D > GM.D_LO).all() and (D < GM.D_HI).all())
        assert bool((r_eq > GM.R_EQ_LO).all() and (r_eq < GM.R_EQ_HI).all())

    def test_map_is_nonlinear_not_affine(self, org):
        """DESIGN.md §2A requires a genuinely nonlinear genome→θ map. An affine
        map would satisfy f(a+b) − f(a) − f(b) + f(0) = 0 identically; this one
        must not."""
        f = lambda a: PL.theta_of(a, org)
        a = jnp.array([0.5, -0.3, 0.2, 0.7])
        b = jnp.array([-0.2, 0.6, -0.4, 0.1])
        defect = f(a + b) - f(a) - f(b) + f(jnp.zeros(N_GENES))
        assert float(jnp.abs(defect).max()) > 1e-3

    def test_bounds_hold_for_extreme_genomes(self, org):
        """The bound is what keeps development inside the regime where ∂x*/∂θ
        exists — it must hold for any genome evolution might propose, not just
        small ones."""
        for v in (-50.0, 50.0):
            D, r_eq = M.unpack_theta(PL.theta_of(jnp.full((N_GENES,), v), org))
            assert bool((D >= GM.D_LO).all() and (D <= GM.D_HI).all())
            assert bool((r_eq >= GM.R_EQ_LO).all() and (r_eq <= GM.R_EQ_HI).all())

    def test_grn_jacobian_matches_finite_differences(self, org, a0):
        J = GM.grn_jacobian(a0, org.coords, org.grn)
        J_fd = FP.finite_difference_sensitivity(
            lambda a: PL.theta_of(a, org), a0, eps=1e-6)
        assert float(jnp.abs(J - J_fd).max()) < 1e-7


# ---------------------------------------------------------------------------
# Layer C — landmarks and Procrustes
# ---------------------------------------------------------------------------

class TestProcrustes:
    def test_shape_is_invariant_to_rigid_motion_and_scale(self, org, a0):
        x = PL.develop(a0, org)
        z = PH.procrustes_shape(x, org.idx, org.ref)

        c, s = jnp.cos(0.7), jnp.sin(0.7)
        R = jnp.array([[c, -s], [s, c]])
        moved = 3.1 * (x @ R.T) + jnp.array([-2.0, 5.0])   # rotate, scale, shift
        z_moved = PH.procrustes_shape(moved, org.idx, org.ref)
        assert float(jnp.abs(z - z_moved).max()) < 1e-9

    def test_alignment_recovers_a_known_rotation(self):
        L = jnp.array([[1.0, 0.0], [0.2, 0.9], [-0.8, 0.1], [0.1, -1.1]])
        ref = PH._centre_and_scale(L)
        c, s = jnp.cos(0.5), jnp.sin(0.5)
        spun = L @ jnp.array([[c, -s], [s, c]]).T
        back = PH.procrustes_align(PH._centre_and_scale(spun), ref)
        assert float(jnp.abs(back - ref).max()) < 1e-9

    def test_shape_dim_is_2k_minus_4(self):
        assert PH.shape_dim(19) == 34

    def _spectrum(self, org, eps, seed=0):
        """Normalised singular values of a cloud of shapes at perturbation
        scale ``eps`` about the reference."""
        rng = np.random.default_rng(seed)
        ref = np.asarray(org.ref)
        k = ref.shape[0]
        L = jnp.asarray(ref[None] + rng.normal(0, eps, (300, k, 2)))
        Z = jnp.stack([PH.procrustes_align(PH._centre_and_scale(l), org.ref).ravel()
                       for l in L])
        s = np.linalg.svd(np.asarray(Z - Z.mean(0)), compute_uv=False)
        return s / s[0]

    def test_shape_space_is_degenerate_by_four_three_exactly_one_asymptotically(
            self, org):
        """``shape_dim`` = 2k−4, but the four lost dimensions are not lost the
        same way, and the difference is visible in the spectrum.

        **Three go exactly**: the two centring conditions and Procrustes
        optimality (Σ zᵢ × refᵢ = 0) are *linear* in ``z``, so their singular
        values sit at machine zero for any cloud.

        **The fourth goes only in the limit**: unit scale is *nonlinear*.
        Shapes lie on a sphere, and at finite spread a cloud still pokes out of
        the tangent plane by O(ε²), leaving a singular value of relative size
        O(ε) rather than zero. So the finite-ε linear rank is 2k−3, and 2k−4 is
        the ε→0 tangent dimension — which is the regime G actually lives in.
        """
        k = org.idx.shape[0]
        s = self._spectrum(org, 1e-3)
        assert len(s) == 2 * k

        # the three exact linear constraints
        assert (s[2 * k - 3:] < 1e-10).all()
        # the 34 genuine shape dimensions are O(1)
        assert s[2 * k - 5] > 1e-1
        # the fourth is suppressed, but not to machine zero
        assert 1e-8 < s[2 * k - 4] < 1e-2
        assert PH.shape_dim(k) == 2 * k - 4

    def test_the_fourth_dimension_vanishes_first_order_in_the_spread(self, org):
        """Pins *why* the fourth direction is a scale artefact and not a real
        shape dimension: its singular value is O(ε), so halving the spread
        halves it, and it disappears in the tangent limit. A genuine dimension
        would keep its O(1) relative size."""
        k = org.idx.shape[0]
        big = self._spectrum(org, 1e-3)[2 * k - 4]
        small = self._spectrum(org, 5e-4)[2 * k - 4]
        assert 0.4 < small / big < 0.6, f"expected O(eps): ratio {small/big:.3f}"

    def test_centroid_size_is_the_scale_that_was_removed(self, org, a0):
        x = PL.develop(a0, org)
        L = PH.landmarks(x, org.idx)
        assert abs(float(PH.centroid_size(2.5 * L) / PH.centroid_size(L)) - 2.5) < 1e-9


class TestReadoutAnnihilatesRigidModes:
    def test_shape_jacobian_kills_translations_and_rotation(self, org, a0):
        """The mechanism behind the gauge-invariance of the chain: the readout
        cannot see a rigid motion, so the gauge the implicit solve picked cannot
        reach the phenotype."""
        x = PL.develop(a0, org)
        dz_dx = PH.shape_jacobian(x, org.idx, org.ref)
        Z = FP.rigid_modes(x, org.alive)
        assert float(jnp.abs(dz_dx @ Z).max()) < 1e-8


# ---------------------------------------------------------------------------
# The composed chain — the Phase 1 payoff
# ---------------------------------------------------------------------------

class TestGaugeInvarianceOfTheChain:
    def test_composed_jacobian_matches_fd_with_no_projection(self, org, a0):
        """Phase 1's gate needed a gauge projection, because raw ∂x*/∂θ carries
        a path-dependent rotation that no fixed-point method reproduces. Compose
        the Procrustes readout on top and that contamination is gone: ∂z/∂a
        matches finite differences **raw**.

        Phase 1's raw comparison failed at ~0.7 relative. This one passes at
        ~1e-8 — same engine, same implicit solve, with a readout that quotients
        out the anholonomy.
        """
        J = PL.phenotype_jacobian(a0, org)
        J_fd = FP.finite_difference_sensitivity(
            lambda a: PL.phenotype(a, org), a0, eps=1e-6)
        rel = float(jnp.linalg.norm(J - J_fd) / jnp.linalg.norm(J_fd))
        assert rel < 1e-6, f"raw (unprojected) chain mismatch: {rel:.3e}"

    def test_chain_sensitivity_is_nontrivial(self, org, a0):
        J = PL.phenotype_jacobian(a0, org)
        assert float(jnp.abs(J).max()) > 1e-4

    def test_cg_and_pinv_agree_through_the_chain(self, org, a0):
        Jp = PL.phenotype_jacobian(a0, org, solver="pinv")
        Jc = PL.phenotype_jacobian(a0, org, solver="cg")
        assert float(jnp.linalg.norm(Jc - Jp) / jnp.linalg.norm(Jp)) < 1e-6


# ---------------------------------------------------------------------------
# Population path
# ---------------------------------------------------------------------------

class TestPopulation:
    def test_batched_development_matches_serial(self, org):
        A = jnp.array([[0.0, 0.0, 0.0, 0.0],
                       [0.01, -0.02, 0.005, 0.0],
                       [-0.03, 0.01, 0.02, -0.01]])
        Z_batch = PL.phenotype_population(A, org)
        Z_serial = jnp.stack([PL.phenotype(a, org) for a in A])
        assert float(jnp.abs(Z_batch - Z_serial).max()) < 1e-9

    def test_population_reports_failure_rather_than_returning_garbage(self, org):
        with pytest.raises(RuntimeError, match="failed to reach equilibrium"):
            PL.phenotype_population(jnp.zeros((2, N_GENES)), org, tol=1e-30)
