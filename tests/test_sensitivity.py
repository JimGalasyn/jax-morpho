"""Phase-0b calibration: validate the developmental-sensitivity engine on the
Milocco & Uller toggle switch, where the answer is known.

Two checks:
  (1) the three sensitivity methods (forward-mode autodiff, reverse-mode
      autodiff, implicit-diff at the steady state) agree — so our autodiff/
      implicit Jacobian is the correct developmental sensitivity;
  (2) sensitivity x allelic-effect equals the Fisher regression average effect
      (their Fig 1C: alpha = s * gamma).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jax_morpho.evodevo.reference_mu import develop_theta, toggle_deriv
from jax_morpho.evodevo.sensitivity import (
    forward_sensitivity, implicit_sensitivity, reverse_sensitivity,
)

# reach steady state; coarse-but-sufficient integration (sensitivity of the
# fixed point is step-size independent).
_dev = lambda th: develop_theta(th, 0.0, 1000, 0.05)
THETA_REF = jnp.array([0.0, 0.0])


class TestThreeWayAgreement:
    def test_forward_reverse_implicit_agree(self):
        Jf = np.asarray(forward_sensitivity(_dev, THETA_REF))
        Jr = np.asarray(reverse_sensitivity(_dev, THETA_REF))
        x_star = _dev(THETA_REF)
        Ji = np.asarray(implicit_sensitivity(
            lambda x, th: toggle_deriv(x, th, 0.0), x_star, THETA_REF))
        # forward and reverse autodiff differentiate the same computation.
        assert np.abs(Jf - Jr).max() < 1e-3
        # implicit-diff at the equilibrium matches the through-solve gradient.
        assert np.abs(Jf - Ji).max() < 1e-3
        # the sensitivity is nontrivial (not accidentally zero).
        assert np.abs(Ji).max() > 0.1


class TestFig1C:
    def test_average_effect_equals_sensitivity_times_gamma(self):
        # single locus affecting theta1; regression average effect on trait 1
        # should equal s1 * gamma (small-perturbation regime).
        J = np.asarray(implicit_sensitivity(
            lambda x, th: toggle_deriv(x, th, 0.0), _dev(THETA_REF), THETA_REF))
        s1 = J[0, 0]                             # d(trait1)/d(theta1)

        dev_batch = jax.jit(jax.vmap(lambda t1: _dev(jnp.stack([t1, 0.0]))))
        rng = np.random.default_rng(0)
        p, n, gamma = 0.5, 3000, 0.01
        g = rng.choice([-1, 0, 1], n, p=[(1 - p) ** 2, 2 * p * (1 - p), p ** 2])
        x1 = np.asarray(dev_batch(jnp.asarray((g * gamma).astype(np.float32))))[:, 0]
        alpha_reg = np.cov(x1, g)[0, 1] / np.var(g, ddof=1)   # match np.cov's ddof
        alpha_sens = s1 * gamma
        assert abs(alpha_reg / alpha_sens - 1.0) < 0.02
