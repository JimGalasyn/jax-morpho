"""Calibration tests: genome -> mechanics -> form, differentiable to the genome.

Distinct genomes must produce distinct, correctly-oriented forms, gradients
must reach the genome, and a target form must be recoverable as the genome
that produces it (gradient-based evo-devo).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jax_morpho.genome import (
    axial_sigmas, develop, fit_genome, genome_morphology,
    sequence_to_genome,
)
from jax_morpho.inverse_design import relax_in_field

N, RELAX = 150, 120


def _disk(seed=0):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, N)
    rad = np.sqrt(rng.uniform(0, 1, N)) * np.sqrt(N) * 0.55
    return jnp.asarray(np.column_stack(
        [rad * np.cos(ang), rad * np.sin(ang)]).astype(np.float32))


def _developed_sigmas(genome, pos0):
    return axial_sigmas(relax_in_field(pos0, develop(genome), RELAX))


class TestGenomeToForm:
    def test_distinct_genomes_distinct_and_oriented_forms(self):
        pos0 = _disk(0)
        # F (hydrophobic) tightens its axis, K (hydrophilic) loosens it.
        sx_fk, sy_fk = _developed_sigmas(sequence_to_genome("F", "K"), pos0)
        sx_kf, sy_kf = _developed_sigmas(sequence_to_genome("K", "F"), pos0)
        sx_gg, sy_gg = _developed_sigmas(sequence_to_genome("G", "G"), pos0)
        # F,K -> tight x, loose y -> elongated along y
        assert float(sy_fk) > float(sx_fk) * 1.15
        # K,F -> the opposite orientation
        assert float(sx_kf) > float(sy_kf) * 1.15
        # G,G -> neutral, roughly round
        assert abs(float(sx_gg) - float(sy_gg)) < 0.25 * float(sx_gg)


class TestGradientReachesGenome:
    def test_gradient_nonzero_and_finite(self):
        pos0 = _disk(1)

        def loss(g):
            sz, ap = genome_morphology(g, pos0, RELAX)
            return (sz - 5.0) ** 2 + (ap - 1.8) ** 2

        g = jnp.array([0.2, -0.2], dtype=jnp.float32)
        grad = jax.grad(loss)(g)
        assert np.all(np.isfinite(np.asarray(grad)))
        assert float(jnp.abs(grad).sum()) > 0.0


class TestEvoDevoInverseDesign:
    def test_recovers_genome_for_target_form(self):
        pos0 = _disk(0)
        g_true = jnp.array([1.0, -1.0], dtype=jnp.float32)   # ~ (F, K)
        size_t, aspect_t = (float(v) for v in
                            genome_morphology(g_true, pos0, RELAX))
        _, history = fit_genome(pos0, size_t, aspect_t,
                                lr=0.08, n_steps=35, relax_steps=RELAX)
        assert history[-1] < 0.1 * history[0], (
            f"loss must drop >=10x: {history[0]:.4f} -> {history[-1]:.5f}")
        assert history[-1] < 0.02, f"final loss {history[-1]:.5f} too high"
