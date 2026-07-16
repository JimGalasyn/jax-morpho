"""Gate #2: ``G = J M Jᵀ`` ⟺ the empirical covariance of developed phenotypes,
in the small-perturbation regime (docs/DESIGN.md validation ladder, rung 2).

The claim is narrow and falsifiable: the delta-method G, built from the
implicit-diff developmental Jacobian, must reproduce the covariance a *real*
population of developed individuals actually shows — and the discrepancy must
vanish as the genetic spread shrinks, because that is the only regime where a
linearisation is entitled to be right.

Method note — common random numbers. The same ``ξ`` is reused at every σ, and
``G`` is built from the **empirical** covariance of the drawn genomes rather
than the nominal ``M``. Both estimators then share their sampling fluctuation,
which cancels to leading order, so what remains is the map's nonlinearity rather
than Monte-Carlo noise.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_morpho.evodevo import genome_map as GM
from jax_morpho.evodevo import mechanical as M
from jax_morpho.evodevo import phenotype as PH
from jax_morpho.evodevo import pipeline as PL
from jax_morpho.evodevo import quantgen as QG

N_GENES = 4
N_SAMPLES = 400


@pytest.fixture(scope="module")
def setup():
    grn = GM.init_grn(jax.random.key(0), N_GENES, hidden=16, scale=1.5)
    a0 = jnp.zeros(N_GENES)
    org = PL.make_organism(grn, a0, n_rings=2)
    return dict(org=org, a0=a0,
                J=PL.phenotype_jacobian(a0, org),
                xi=jax.random.normal(jax.random.key(7), (N_SAMPLES, N_GENES)))


def _gate(setup, sigma):
    """Relative difference between the delta-method G and the empirical
    covariance of a developed population at genetic spread ``sigma``."""
    A = setup["a0"] + sigma * setup["xi"]
    Z = PL.phenotype_population(A, setup["org"])
    G_lin = QG.build_G(setup["J"], QG.empirical_covariance(A))
    return QG.relative_difference(QG.empirical_covariance(Z), G_lin)


class TestGate2:
    def test_G_matches_empirical_covariance_at_small_sigma(self, setup):
        """The gate. ~1.8e-03 as measured."""
        rel = _gate(setup, 1.25e-3)
        assert rel < 5e-3, f"gate #2 failed: relative difference {rel:.3e}"

    def test_discrepancy_shrinks_with_the_perturbation(self, setup):
        """The delta method is a *local* claim, so its error must be controlled
        by σ. A G that matched equally well at every σ would be matching for the
        wrong reason."""
        big, small = _gate(setup, 1e-2), _gate(setup, 1.25e-3)
        assert small < big / 3, f"error did not shrink with sigma: {big:.3e} -> {small:.3e}"

    def test_G_is_not_trivially_zero(self, setup):
        G = QG.build_G(setup["J"], jnp.eye(N_GENES))
        assert float(jnp.abs(G).max()) > 1e-8


class TestGStructure:
    def test_rank_is_limited_by_the_genome(self, setup):
        """G inherits J's rank. With 4 genes feeding a 34-dimensional shape
        space, G is rank 4: development cannot express more independent
        directions of variation than the genome supplies. (The other constraint
        — that shape space is itself 2k−4, not 2k — bites once n_genes exceeds
        it; see test_phenotype.py.)"""
        G = QG.build_G(setup["J"], jnp.eye(N_GENES))
        assert np.linalg.matrix_rank(np.asarray(G), tol=1e-12) == N_GENES
        assert N_GENES < PH.shape_dim(setup["org"].idx.shape[0])

    def test_G_is_symmetric_psd(self, setup):
        G = np.asarray(QG.build_G(setup["J"], jnp.eye(N_GENES)))
        assert np.abs(G - G.T).max() < 1e-12
        assert np.linalg.eigvalsh(G).min() > -1e-12


class TestDevelopmentalMultistability:
    """Why gate #2 is a *small*-perturbation claim, and not a technicality.

    The Morse landscape is multistable. Past a threshold genetic perturbation
    the tissue rearranges its neighbours — a T1-like swap — and lands in a
    different packing. The phenotype then jumps discontinuously, and a local
    Jacobian is silent about it: G describes the response *within* a basin.

    This is not an embarrassment for the framework; it is the mechanical analogue
    of the bistability in Milocco & Uller's own reference model, which is a
    *toggle switch*. Their development is multistable by construction; ours turns
    out to be multistable by consequence.
    """

    def _forms(self, setup, sigma, n=120):
        org, a0 = setup["org"], setup["a0"]
        X, _, _ = PL.develop_population(a0 + sigma * setup["xi"][:n], org)
        x0 = PL.develop(a0, org)
        dx = np.array([float(jnp.linalg.norm(x - x0)) for x in X])
        return X, x0, dx

    def test_large_perturbations_jump_to_a_different_packing(self, setup):
        """The finding, on the unambiguous criterion: at σ=0.05 the distribution
        of |x* − x*(a0)| is **bimodal**. Most individuals deform smoothly
        (~0.02); a couple land ~0.43 away — a 20x gap, far too large to be
        deformation, and far too clean to be a threshold artefact."""
        _, _, dx = self._forms(setup, 0.05)
        jumped = dx > 0.2
        assert jumped.sum() > 0, "expected some basin jumps at sigma=0.05"
        assert np.median(dx[jumped]) > 10 * np.median(dx[~jumped])

    def test_no_jumps_in_the_gated_regime(self, setup):
        """...and none at the σ where gate #2 is stated, which is why gate #2 is
        entitled to be a purely local claim there."""
        _, _, dx = self._forms(setup, 1.25e-3)
        assert (dx < 0.2).all()
        assert (M.contact_topology(self._forms(setup, 1.25e-3)[0][0])
                == M.contact_topology(PL.develop(setup["a0"], setup["org"])))

    def test_jumps_are_real_rearrangement_not_procrustes_drift(self, setup):
        """Rules out the obvious alternative. A near-C6-symmetric blob could in
        principle flip Procrustes alignment between symmetry-equivalent optima,
        which would look like a jump. It does not: alignment angles stay under a
        tenth of a degree across the whole population, while the jumping
        individuals move the *unaligned* equilibrium 20x further than anyone else
        and change which cells touch."""
        org = setup["org"]
        X, _, dx = self._forms(setup, 0.05)
        jumped = dx > 0.2
        assert jumped.any()

        angles = []
        for x in X:
            L = PH._centre_and_scale(PH.landmarks(x, org.idx))
            S = (L[:, 0] * org.ref[:, 1] - L[:, 1] * org.ref[:, 0]).sum()
            C = (L * org.ref).sum()
            angles.append(abs(float(jnp.degrees(jnp.arctan2(S, C)))))
        assert max(angles) < 1.0          # no symmetry-branch flip anywhere

        # every jumper genuinely rearranged its contacts
        ref_contacts = M.contact_topology(PL.develop(setup["a0"], org))
        for x in np.asarray(X)[jumped]:
            assert M.contact_topology(x) != ref_contacts

    def test_delaunay_would_be_the_wrong_fingerprint_here(self, setup):
        """Pins why :func:`contact_topology` is contact-based. A hex lattice is
        maximally cocircular, so its Delaunay triangulation flips diagonals under
        infinitesimal perturbation with nothing physical changing — it reports
        far more 'rearrangements' than actually occur."""
        from scipy.spatial import Delaunay

        def edges(P):
            t = Delaunay(np.asarray(P))
            return frozenset((int(min(s[i], s[j])), int(max(s[i], s[j])))
                             for s in t.simplices
                             for i in range(3) for j in range(i + 1, 3))

        X, x0, dx = self._forms(setup, 0.05)
        jumped = dx > 0.2
        e0 = edges(x0)
        flagged = np.array([edges(x) != e0 for x in X])
        # Delaunay flags an order of magnitude more than really jumped.
        assert flagged.sum() > 5 * max(jumped.sum(), 1)
        # while contacts stay far closer to the truth.
        c0 = M.contact_topology(x0)
        c_flag = np.array([M.contact_topology(x) != c0 for x in X])
        assert c_flag.sum() < flagged.sum()
