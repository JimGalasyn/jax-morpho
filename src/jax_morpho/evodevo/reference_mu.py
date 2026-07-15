"""Phase-0 calibration anchor: a faithful port of Milocco & Uller (2026 PNAS)
"Bridging developmental and statistical approaches to variation and evolution",
Figure 3C.

This is *their* model, reproduced exactly so that we have a verified reference
before swapping in our own (mechanical) developmental engine. Source of truth:
their MATLAB code at github.com/lisandromilocco/DevStat-Bridge (Figure3/).

Their development is a bistable toggle-switch gene network (2 coupled ODEs);
the phenotype is the pair of expression levels at developmental time t=50. A
population's additive-genetic covariance G is built from *regression* average
effects, and the one-generation response to truncation selection toward an
optimum is compared with the multivariate breeder's equation using G vs. the
phenotypic covariance P. The paper's result: G predicts the response (small
angle) while P misaligns at low minor-allele frequency.

Only the development is in JAX (so it is autodiff-ready for the Phase-0b
sensitivity cross-check and vmaps over the population); the quantitative-
genetics bookkeeping stays in NumPy, matching their scripts line for line.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

# --- their model constants (sistemaggT.m / LandePrediction_Generate.m) ---
T_END = 50.0
DT = 0.01
G0 = (0.6, 0.8)              # initial condition Xio
N_LOCI = 10                 # loci per developmental parameter
SIGMA_GAMMA = 1e-4          # SD of per-locus allelic effect
SIGMA_ENV = 1.5e-3          # SD of the environmental parameter u (their lambda_3)
OPTIMUM = (4.0, 4.0)


# ---------------------------------------------------------------------------
# Development: the toggle-switch ODE, RK4 to steady state (t=50)
# ---------------------------------------------------------------------------

def _deriv(g, th1, th2, u):
    g1, g2 = g[0], g[1]
    dg1 = (2.0 + th1) / (1.0 + (g2 / 2.0) ** 2) - 0.4 * g1
    dg2 = (2.0 + th2) / (1.0 + (g1 / (3.0 + u)) ** 2) - 0.4 * g2
    return jnp.stack([dg1, dg2])


@partial(jax.jit, static_argnums=(3,))
def develop(th1, th2, u, n_steps=int(T_END / DT), dt=DT):
    """Integrate the toggle switch to t=50; return phenotype (g1, g2) there.

    th1, th2, u are per-individual scalars (constant over development, as in
    their Fig 3C). Fixed-step RK4 — the system reaches a steady state well
    before t=50, so this matches their adaptive ode45 endpoint.
    """
    def step(g, _):
        k1 = _deriv(g, th1, th2, u)
        k2 = _deriv(g + 0.5 * dt * k1, th1, th2, u)
        k3 = _deriv(g + 0.5 * dt * k2, th1, th2, u)
        k4 = _deriv(g + dt * k3, th1, th2, u)
        return g + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None

    g, _ = jax.lax.scan(step, jnp.array(G0), None, length=n_steps)
    return g


_develop_pop = jax.jit(jax.vmap(lambda th1, th2, u: develop(th1, th2, u)))


def develop_population(th1, th2, u):
    """Vectorized development of a whole population. Arrays of shape (N,)."""
    z = _develop_pop(jnp.asarray(th1), jnp.asarray(th2), jnp.asarray(u))
    return np.asarray(z)          # (N, 2) phenotypes (g1, g2) at t=50


# ---------------------------------------------------------------------------
# Population genetics (NumPy, matching their MATLAB line for line)
# ---------------------------------------------------------------------------

def sample_genotypes(p, n_loci, n_ind, rng):
    """Genotype scores {-1,0,1} at HWE freqs [q^2, 2pq, p^2] per locus."""
    q = 1.0 - p
    probs = [q * q, 2 * p * q, p * p]
    return rng.choice([-1, 0, 1], size=(n_loci, n_ind), p=probs)


def regression_average_effects(gscore1, gscore2, z):
    """Fisher average effects via least-squares regression of centered
    phenotype on genotype scores (their Z \\ dx). Returns (2*n_loci+1, 2)."""
    n = z.shape[0]
    Z = np.column_stack([gscore1.T, gscore2.T, np.ones(n)])   # (N, 2L+1)
    dz = z - z.mean(0)
    alpha, *_ = np.linalg.lstsq(Z, dz, rcond=None)            # (2L+1, 2)
    return alpha


def build_G(alpha, genotype):
    """G = sum_i 2 p_i q_i alpha_i alpha_i^T over the loci (their loop),
    with p_i,q_i from realized genotype counts. genotype: (N, 2L)."""
    n_loci = genotype.shape[1]
    G = np.zeros((2, 2))
    for i in range(n_loci):
        col = genotype[:, i]
        n_A2 = np.sum(col == 1) * 2 + np.sum(col == 0)
        n_A1 = np.sum(col == -1) * 2 + np.sum(col == 0)
        tot = n_A1 + n_A2
        p_i, q_i = n_A2 / tot, n_A1 / tot
        a = alpha[i][:, None]                                 # (2,1)
        G += (a @ a.T) * 2 * p_i * q_i
    return G


def recombine(a1, a2, rng):
    """Their recombine(): one offspring allele from two parental alleles,
    no linkage (Mendelian)."""
    if a1 == 1 and a2 == 1:
        return 1
    if a1 == -1 and a2 == -1:
        return -1
    if {a1, a2} == {0, -1}:
        return rng.choice([-1, 0])
    if {a1, a2} == {0, 1}:
        return rng.choice([0, 1])
    if {a1, a2} == {-1, 1}:
        return 0
    if a1 == 0 and a2 == 0:
        return rng.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
    raise ValueError(f"unexpected alleles {a1}, {a2}")


def recombine_vec(parentA, parentB, n_off, rng):
    """Vectorized Mendelian recombination, exactly equivalent to recombine():
    a {-1,0,1} score is a +allele count minus 1, so a parent transmits a gamete
    ~ Bernoulli(count/2); offspring score = gameteA + gameteB - 1. Produces
    n_off offspring per parent pair. parentA/B: (n_loci, n_pairs)."""
    ca = np.repeat(parentA, n_off, axis=1) + 1        # +allele count 0..2
    cb = np.repeat(parentB, n_off, axis=1) + 1
    gA = rng.binomial(1, ca / 2.0)
    gB = rng.binomial(1, cb / 2.0)
    return gA + gB - 1                                 # (n_loci, n_pairs*n_off)


# ---------------------------------------------------------------------------
# Figure 3C protocol
# ---------------------------------------------------------------------------

def angle_deg(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def simulate_fig3c(p2, p1=0.5, n_ind=5000, n_replays=50, dt=DT, seed=0):
    """One point of Fig 3C: build G and P for minor-allele-freq p2 (on the
    theta2 loci; theta1 loci fixed at p1=0.5), apply one round of truncation
    selection toward the optimum, and compare the multivariate breeder's
    prediction (G-based vs P-based) with the observed recombinant response."""
    rng = np.random.default_rng(seed)
    n_steps = int(T_END / dt)

    gamma1 = rng.normal(0, SIGMA_GAMMA, N_LOCI)
    gamma2 = rng.normal(0, SIGMA_GAMMA, N_LOCI)
    gs1 = sample_genotypes(p1, N_LOCI, n_ind, rng)     # (L, N)
    gs2 = sample_genotypes(p2, N_LOCI, n_ind, rng)
    th1 = (gs1 * gamma1[:, None]).sum(0)
    th2 = (gs2 * gamma2[:, None]).sum(0)
    u = rng.normal(0, SIGMA_ENV, n_ind)

    dev = lambda a, b, c: np.asarray(
        jax.jit(jax.vmap(lambda x, y, z: develop(x, y, z, n_steps, dt)))(
            jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)))
    z = dev(th1, th2, u)                               # (N, 2) phenotypes

    alpha = regression_average_effects(gs1, gs2, z)    # (2L+1, 2)
    genotype = np.column_stack([gs1.T, gs2.T])         # (N, 2L)
    G = build_G(alpha[:-1], genotype)                  # drop intercept row
    P = np.cov((z - z.mean(0)).T)

    dist = np.sqrt(((z - np.array(OPTIMUM)) ** 2).sum(1))
    sel = np.argsort(dist)[: n_ind // 2]
    mean_all = z.mean(0)
    s = z[sel].mean(0) - mean_all
    beta = np.linalg.solve(P, s)
    dZ_lande = G @ beta
    dZ_naive = s.copy()

    par1, par2 = gs1[:, sel], gs2[:, sel]
    n_sel = len(sel)
    obs = []
    for _ in range(n_replays):
        perm = rng.permutation(n_sel)
        pairs = perm[: 2 * (n_sel // 2)].reshape(-1, 2)
        off1 = recombine_vec(par1[:, pairs[:, 0]], par1[:, pairs[:, 1]], 4, rng)
        off2 = recombine_vec(par2[:, pairs[:, 0]], par2[:, pairs[:, 1]], 4, rng)
        th1o = (off1 * gamma1[:, None]).sum(0)
        th2o = (off2 * gamma2[:, None]).sum(0)
        uo = rng.normal(0, SIGMA_ENV, off1.shape[1])
        zo = dev(th1o, th2o, uo)
        obs.append(zo.mean(0) - mean_all)
    dZ_obs = np.array(obs).mean(0)

    return dict(p2=p2, G=G, P=P, s=s, dZ_lande=dZ_lande, dZ_naive=dZ_naive,
                dZ_obs=dZ_obs,
                angle_G=angle_deg(dZ_obs, dZ_lande),
                angle_P=angle_deg(dZ_obs, dZ_naive))


def run_fig3c(n_ind=5000, n_replays=50, dt=DT, seed=0):
    """Sweep the theta2 minor-allele frequency over 0.5/2^k (k=9..0), matching
    their mu_values, and return the per-p results."""
    p2_values = 0.5 / 2.0 ** np.arange(9, -1, -1)
    return [simulate_fig3c(float(p2), n_ind=n_ind, n_replays=n_replays,
                           dt=dt, seed=seed + k)
            for k, p2 in enumerate(p2_values)]

