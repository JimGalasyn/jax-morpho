"""Phase-0a calibration: our port must reproduce Milocco & Uller (2026) Fig 3C.

The load-bearing claim: the development-derived G predicts the one-generation
response to selection across allele frequencies, while the phenotypic
covariance P misaligns at low minor-allele frequency. We assert the robust
qualitative pattern (generous margins, so CPU/GPU float differences don't make
it flaky), not their exact numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from jax_morpho.evodevo.reference_mu import (
    develop, recombine, recombine_vec, simulate_fig3c,
)


class TestDevelopment:
    def test_steady_state_finite_and_deterministic(self):
        a = np.asarray(develop(0.0, 0.0, 0.0))
        b = np.asarray(develop(0.0, 0.0, 0.0))
        assert np.isfinite(a).all()
        assert np.allclose(a, b)                 # deterministic
        assert (a > 0).all() and (a < 20).all()  # sensible expression levels


class TestRecombination:
    def test_vectorized_matches_mendelian_expectation(self):
        # For each parental score pair, the offspring-score distribution from
        # recombine_vec must match many draws of the scalar recombine().
        rng = np.random.default_rng(0)
        for a1 in (-1, 0, 1):
            for a2 in (-1, 0, 1):
                pa = np.full((1, 4000), a1)
                pb = np.full((1, 4000), a2)
                vec = recombine_vec(pa, pb, 1, rng).ravel()
                ref = np.array([recombine(a1, a2, rng) for _ in range(4000)])
                for s in (-1, 0, 1):
                    assert abs((vec == s).mean() - (ref == s).mean()) < 0.05


class TestFig3CCalibration:
    @pytest.fixture(scope="class")
    def low_freq(self):
        return simulate_fig3c(0.001, n_ind=1500, n_replays=18, dt=0.05, seed=0)

    @pytest.fixture(scope="class")
    def high_freq(self):
        return simulate_fig3c(0.5, n_ind=1500, n_replays=18, dt=0.05, seed=0)

    def test_P_fails_at_low_frequency(self, low_freq):
        # G predicts; P misaligns badly at low MAF (their Fig 3C result).
        assert low_freq["angle_P"] > low_freq["angle_G"] + 10.0
        assert low_freq["angle_P"] > 20.0

    def test_G_and_P_proportional_at_high_frequency(self, high_freq):
        # At MAF 0.5, G ≈ P and both predict well.
        assert high_freq["angle_G"] < 8.0
        assert high_freq["angle_P"] < 8.0
        assert abs(high_freq["angle_G"] - high_freq["angle_P"]) < 4.0

    def test_G_predicts_at_both_frequencies(self, low_freq, high_freq):
        # The development-derived G stays a reasonable predictor throughout.
        assert low_freq["angle_G"] < 30.0
        assert high_freq["angle_G"] < 8.0


class TestPhase0cSensitivityG:
    """Phase 0c: G built from *our* autodiff/implicit sensitivity (alpha =
    gamma * s) must equal their regression G and predict the response as well."""

    @pytest.fixture(scope="class")
    def low(self):
        return simulate_fig3c(0.001, n_ind=2000, n_replays=12, dt=0.05, seed=0)

    @pytest.fixture(scope="class")
    def high(self):
        return simulate_fig3c(0.5, n_ind=2000, n_replays=12, dt=0.05, seed=0)

    def test_sensitivity_G_matches_regression_G(self, low, high):
        assert high["G_rel_frob"] < 0.10
        assert low["G_rel_frob"] < 0.20

    def test_sensitivity_G_predicts_the_response(self, low, high):
        assert high["angle_G_sens"] < 8.0
        assert low["angle_G_sens"] < 15.0
        # our G predicts about as well as their regression G
        assert abs(low["angle_G_sens"] - low["angle_G"]) < 6.0

    def test_sensitivity_G_beats_P_at_low_frequency(self, low):
        assert low["angle_P"] > low["angle_G_sens"] + 8.0
