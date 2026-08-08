"""State-of-the-art recurrent gene-regulatory network: the Wagner model.

This is the standard GRN model of evolutionary-systems biology, introduced by
Andreas Wagner (1996, *Evolution* 50:1008, "Does evolutionary plasticity
evolve?") and used by Siegal & Bergman (2002, *PNAS* 99:10528, "Waddington's
canalization revisited") to show that stabilizing selection on *development*
spontaneously produces mutational robustness — canalization — with no direct
selection for robustness itself.

The model, verbatim:

    s_{t+1} = sigma(W . s_t),     sigma(x) = tanh(a*x/2) = 2/(1+e^{-a*x}) - 1

  * s in (-1, 1)^N : expression state of N genes (-1 fully off, +1 fully on).
  * W in R^{N x N} : the regulatory matrix; W[i, j] is the effect of gene j on
                     gene i. A fraction `density` of entries are nonzero, drawn
                     from N(0, 1). **This matrix is the genome.**
  * a              : activation steepness (Siegal & Bergman's `a`).

Development runs the map from a fixed founder initial state `s0` for up to
`devsteps` steps. A network is *developmentally stable* (viable) when the state
settles: the windowed variance xi over the last `window` steps falls below
`eps` (Siegal & Bergman's xi < 1e-4). The settled expression pattern is the
phenotype.

Relation to the rest of `evodevo`
---------------------------------
This is a *discrete gene-network* model and is deliberately NOT the same object
as `evodevo.genome_map.GRN`, which is a continuous spatial field. Both are
called "GRN" in the literature; they are not interchangeable, and for that
reason this module is **not re-exported** from `jax_morpho.evodevo`. Import it
by name:

    from jax_morpho.evodevo import wagner_grn

In particular `wagner_grn.develop` is a different function from the package's
`develop_genome` / `develop_mechanical`, and flattening it into the package
namespace would make import order decide which one a caller gets.

Because development here is a `lax.scan`, the phenotype is differentiable in W
for free; the implicit-function route in `jax_morpho.evodevo.fixed_point`
(``fixed_point_sensitivity``) is the O(1)-memory alternative once N is large and
you want gradients through the true equilibrium rather than a fixed unroll.

Everything is pure JAX: use `develop_pop` (a jitted `vmap` of `develop`) to run
an entire population in one batched call — no Python loop over individuals. Do
not rebuild that batcher at the call site; it already exists here.

Determinism on GPU — READ BEFORE TRUSTING A RESULT
---------------------------------------------------
`evolve` (and anything downstream of it) can return **different results across
processes on GPU**, at identical seeds and identical library versions. The cause
is XLA autotuning selecting different kernels per process. It is not a bug in
this module, and it does not show up on CPU.

The consequence is not academic: a stabilizing-vs-drift contrast measured this
way once flipped sign between runs.

Two mitigations, and you want one of them before quoting any number:

  * set ``--xla_gpu_autotune_level=0`` in ``XLA_FLAGS`` (see
    `deterministic_xla_flags` below), or
  * run on CPU, which is the recommended way to validate any refactor of this
    module — a CPU A/B is exact, a GPU A/B is not evidence.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------- #
# Hyperparameters (Siegal & Bergman 2002 defaults where they name one)
# --------------------------------------------------------------------------- #

DEFAULT_N = 10          # genes per network (S&B used 10)
DEFAULT_DENSITY = 0.75  # fraction of nonzero regulatory interactions
DEFAULT_A = 6.0         # activation steepness; tanh(a*x/2). Wagner's original
                        # used a Heaviside step (a -> inf); canalization needs a
                        # steep-enough sigmoid that phenotypes are actually
                        # mutation-sensitive. a in [6, 10] reproduces S&B; a~1
                        # is too gentle (robustness saturates, no headroom).
DEFAULT_DEVSTEPS = 100  # max developmental iterations
DEFAULT_WINDOW = 10     # trailing window for the stability test
DEFAULT_EPS = 1e-4      # xi < eps  =>  developmentally stable (viable)

# Kernel autotuning is what makes `evolve` vary across processes on GPU.
_AUTOTUNE_OFF = "--xla_gpu_autotune_level=0"


def deterministic_xla_flags(existing: str | None = None) -> str:
    """The ``XLA_FLAGS`` value that makes GPU runs reproducible across processes.

    Returns `existing` with the autotune-disabling flag appended (idempotent, and
    a caller's own flags survive). This must be set **before** the first JAX
    computation is compiled — typically at process start:

        import os
        from jax_morpho.evodevo import wagner_grn
        os.environ["XLA_FLAGS"] = wagner_grn.deterministic_xla_flags(
            os.environ.get("XLA_FLAGS"))

    Deliberately a value you install yourself rather than an import-time side
    effect: a library that mutates the environment on import breaks any caller
    who set those flags for their own reasons, and would do so invisibly.

    No effect on CPU, where results are already reproducible.
    """
    existing = existing or ""
    if _AUTOTUNE_OFF in existing:
        return existing
    return " ".join(f for f in (existing, _AUTOTUNE_OFF) if f)


def sigma(x: jnp.ndarray, a: float = DEFAULT_A) -> jnp.ndarray:
    """Siegal-Bergman activation: tanh(a*x/2) == 2/(1+e^{-a x}) - 1, in (-1, 1)."""
    return jnp.tanh(0.5 * a * x)


# --------------------------------------------------------------------------- #
# Network construction
# --------------------------------------------------------------------------- #

class Network(NamedTuple):
    """A GRN individual. `W` is the evolving genome; `mask` fixes the topology
    (which interactions exist); `s0` is the fixed founder initial state."""
    W: jnp.ndarray      # (N, N) regulatory matrix
    mask: jnp.ndarray   # (N, N) {0,1} nonzero-interaction topology
    s0: jnp.ndarray     # (N,) founder initial expression


def random_network(key, n: int = DEFAULT_N, density: float = DEFAULT_DENSITY,
                   a: float = DEFAULT_A) -> Network:
    """Draw a random Wagner network: sparse N(0,1) matrix + random +/-1 s0."""
    k_w, k_m, k_s = jax.random.split(key, 3)
    W_full = jax.random.normal(k_w, (n, n))
    mask = (jax.random.uniform(k_m, (n, n)) < density).astype(W_full.dtype)
    s0 = jnp.where(jax.random.uniform(k_s, (n,)) < 0.5, -1.0, 1.0)
    return Network(W=W_full * mask, mask=mask, s0=s0)


# --------------------------------------------------------------------------- #
# Development (the differentiable core)
# --------------------------------------------------------------------------- #

class DevResult(NamedTuple):
    phenotype: jnp.ndarray  # (N,) settled expression (mean over trailing window)
    stable: jnp.ndarray     # () bool: xi < eps
    xi: jnp.ndarray         # () windowed variance
    s_T: jnp.ndarray        # (N,) FINAL expression state — the resume point

    # `s_T` is appended, not inserted: every consumer in this repo reads
    # DevResult by attribute (checked), so a trailing field is additive. Tuple
    # unpacking would break, and that is why it goes last.
    #
    # Why it exists: `develop` already takes `s0`, so resuming a partial settle
    # is just passing the previous `s_T` back in — no new machinery, only the
    # state that was being discarded into `_`. Required by
    # `Planktonica/docs/DESIGN_larval.md` §2 for bounded-cost larval development,
    # and named there as the one small upstream change the whole design needs.


@partial(jax.jit, static_argnames=("devsteps", "window"))
def develop(W: jnp.ndarray, s0: jnp.ndarray, a: float = DEFAULT_A,
            devsteps: int = DEFAULT_DEVSTEPS, window: int = DEFAULT_WINDOW,
            eps: float = DEFAULT_EPS) -> DevResult:
    """Iterate s_{t+1} = sigma(W s_t) for `devsteps`; report stability + phenotype.

    Differentiable in W (the scan unrolls the recurrence). Faithful to
    Siegal & Bergman: stability is windowed variance over the last `window`
    states falling below `eps`.

    Resuming
    --------
    `s0` *is* the expression state, so a partial settle resumes by feeding the
    previous call's `s_T` back in as `s0`. Splitting `devsteps` into chunks
    reproduces the unsplit trajectory exactly.

    **What composes and what does not.** `s_T` composes over any chunking.
    `phenotype`, `xi` and `stable` do NOT in general: they are statistics over
    the trailing `window` of *this call's* trajectory, so a chunk shorter than
    `window` reports a window it never had. They agree with the unsplit run
    exactly when the final chunk is at least `window` long — which is why
    `DESIGN_larval.md` sets `LARVAL_STEPS = 10` against `DEFAULT_WINDOW = 10`
    rather than picking a smaller step for finer granularity.
    """
    def step(s, _):
        s_next = sigma(W @ s, a)
        return s_next, s_next

    s_T, traj = jax.lax.scan(step, s0, None, length=devsteps)  # (devsteps, N)
    tail = traj[-window:]                                      # (window, N)
    mean = tail.mean(axis=0)                                   # (N,)
    xi = ((tail - mean) ** 2).mean()                           # scalar S&B xi
    return DevResult(phenotype=mean, stable=xi < eps, xi=xi, s_T=s_T)


# Batched development over a whole population (one call, no Python loop).
develop_pop = jax.jit(
    jax.vmap(develop, in_axes=(0, 0, None, None, None, None)),
    static_argnames=("devsteps", "window"),
)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def fitness(phenotype: jnp.ndarray, stable: jnp.ndarray, s_opt: jnp.ndarray,
            sigma_sel: float) -> jnp.ndarray:
    """Stabilizing selection toward `s_opt`, gated on developmental stability.

    Inviable (unstable) networks get fitness 0; viable ones get a Gaussian in
    the distance from the optimum phenotype: F = exp(-||p - s_opt||^2 /
    (2 sigma_sel^2)). With s_opt = the founder's own equilibrium there is no
    directional pressure — selection is purely for *staying put and stable*,
    which is the Siegal-Bergman canalization regime.
    """
    d2 = jnp.sum((phenotype - s_opt) ** 2)
    gaussian = jnp.exp(-d2 / (2.0 * sigma_sel ** 2))
    return jnp.where(stable, gaussian, 0.0)


def viability_fitness(stable: jnp.ndarray) -> jnp.ndarray:
    """Control regime: select for viability only (no phenotype pressure).
    Drift among developmentally stable networks."""
    return stable.astype(jnp.float32)


# --------------------------------------------------------------------------- #
# Mutation
# --------------------------------------------------------------------------- #

def mutate(key, W: jnp.ndarray, mask: jnp.ndarray, rate: float,
           sd: float) -> jnp.ndarray:
    """Per-interaction Gaussian mutation on existing (nonzero-mask) entries.

    Each existing interaction is perturbed independently with probability
    `rate` by N(0, sd). Topology (mask) is conserved, as in Siegal-Bergman
    (mutations change interaction *strengths*, not the wiring).
    """
    k_hit, k_delta = jax.random.split(key)
    hit = (jax.random.uniform(k_hit, W.shape) < rate) * mask
    delta = jax.random.normal(k_delta, W.shape) * sd
    return W + hit * delta


def single_edge_mutation(key, W: jnp.ndarray, mask: jnp.ndarray,
                         sd: float) -> jnp.ndarray:
    """Perturb exactly one existing interaction by N(0, sd).

    The canonical single-step mutational-neighbourhood probe used to measure
    developmental sensitivity: it isolates the effect of one interaction change
    rather than the compound kick of the whole per-generation process.
    """
    k_idx, k_delta = jax.random.split(key)
    flat = mask.reshape(-1)
    idx = jax.random.choice(k_idx, flat.shape[0], p=flat / flat.sum())
    delta = jax.random.normal(k_delta) * sd
    return W.reshape(-1).at[idx].add(delta).reshape(W.shape)


# --------------------------------------------------------------------------- #
# Mutational robustness (the canalization observable)
# --------------------------------------------------------------------------- #

@partial(jax.jit, static_argnames=("n_probes", "devsteps", "window"))
def mutational_robustness(key, W, mask, s0, a=DEFAULT_A, sd=1.0, tau=0.3,
                          n_probes=20, devsteps=DEFAULT_DEVSTEPS,
                          window=DEFAULT_WINDOW, eps=DEFAULT_EPS) -> jnp.ndarray:
    """Buffering of an individual's development against its own mutations.

    Reference is the individual's *own* equilibrium phenotype `p_wt`. For each
    of `n_probes` single-interaction mutations, develop the mutant and score

        r = stable * exp(-||p_mutant - p_wt||^2 / (2 tau^2))     in [0, 1]

    High r means the mutation left development both viable and near the original
    outcome. The mean over probes is the individual's mutational robustness —
    the Siegal-Bergman canalization observable, made continuous (no threshold
    cliff) and self-referenced (independent of any phenotype drift). Returns 0
    for an inviable wild-type.
    """
    wt = develop(W, s0, a, devsteps, window, eps)
    keys = jax.random.split(key, n_probes)

    def probe(k):
        Wm = single_edge_mutation(k, W, mask, sd)
        res = develop(Wm, s0, a, devsteps, window, eps)
        d2 = jnp.sum((res.phenotype - wt.phenotype) ** 2)
        return res.stable.astype(jnp.float32) * jnp.exp(-d2 / (2.0 * tau ** 2))

    probe_scores = jax.vmap(probe)(keys).mean()
    return jnp.where(wt.stable, probe_scores, 0.0)


population_robustness = jax.jit(
    jax.vmap(mutational_robustness,
             in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None)),
    static_argnames=("n_probes", "devsteps", "window"),
)


# --------------------------------------------------------------------------- #
# Viable founder + evolutionary loop
# --------------------------------------------------------------------------- #

def find_viable_founder(key, n=DEFAULT_N, density=DEFAULT_DENSITY, a=DEFAULT_A,
                        devsteps=DEFAULT_DEVSTEPS, window=DEFAULT_WINDOW,
                        eps=DEFAULT_EPS, max_tries=512) -> Network:
    """Rejection-sample a random network that develops to a stable equilibrium.

    Draw a batch, keep the first developmentally stable one. Raises if none of
    `max_tries` candidates is viable (raise density or a).
    """
    keys = jax.random.split(key, max_tries)
    nets = jax.vmap(lambda k: random_network(k, n, density, a))(keys)
    res = develop_pop(nets.W, nets.s0, a, devsteps, window, eps)
    idx = jnp.argmax(res.stable.astype(jnp.int32))  # first stable
    if not bool(res.stable[idx]):
        raise RuntimeError(
            f"no viable founder in {max_tries} tries at density={density}, a={a}"
        )
    return Network(W=nets.W[idx], mask=nets.mask[idx], s0=nets.s0[idx])


class EvolveHistory(NamedTuple):
    generation: list        # sampled generation indices
    robustness: list        # mean population mutational robustness at each
    mean_fitness: list      # mean population fitness at each
    viable_fraction: list   # fraction developmentally stable at each


def founder_optimum(founder: Network, a=DEFAULT_A, devsteps=DEFAULT_DEVSTEPS,
                    window=DEFAULT_WINDOW, eps=DEFAULT_EPS) -> jnp.ndarray:
    """The founder's own equilibrium phenotype — the stabilizing-selection target."""
    return develop(founder.W, founder.s0, a, devsteps, window, eps).phenotype


def founding_population(founder: Network, pop_size: int) -> jnp.ndarray:
    """``pop_size`` copies of the founder's W — generation 0 of a lineage."""
    n = founder.W.shape[0]
    return jnp.broadcast_to(founder.W, (pop_size, n, n)).copy()


def measure_population(key, W, mask, s0, s_opt, *, a=DEFAULT_A, sigma_sel=1.0,
                       probe_sd=1.0, tau=0.3, robustness_probes=20,
                       devsteps=DEFAULT_DEVSTEPS, window=DEFAULT_WINDOW,
                       eps=DEFAULT_EPS) -> dict:
    """The three population observables, measured on ``W`` without advancing it.

    Split out of :func:`evolve`'s inner ``record`` so a campaign RunFn can stream
    the same numbers per generation (`morphospace.runfns`) instead of
    reimplementing the measurement and quietly drifting from it.
    """
    pop_size, n = W.shape[0], W.shape[1]
    res = develop_pop(W, jnp.broadcast_to(s0, (pop_size, n)),
                      a, devsteps, window, eps)
    fit = jax.vmap(lambda p, st: fitness(p, st, s_opt, sigma_sel))(
        res.phenotype, res.stable)
    rob = population_robustness(
        jax.random.split(key, pop_size),
        W, jnp.broadcast_to(mask, (pop_size, n, n)),
        jnp.broadcast_to(s0, (pop_size, n)),
        a, probe_sd, tau, robustness_probes, devsteps, window, eps)
    return {"robustness": float(rob.mean()),
            "mean_fitness": float(fit.mean()),
            "viable_fraction": float(res.stable.mean())}


def evolve_step(k_sel, k_mut, W, mask, s0, s_opt, *, a=DEFAULT_A, sigma_sel=1.0,
                mut_rate=0.02, mut_sd=0.5, stabilizing=True,
                devsteps=DEFAULT_DEVSTEPS, window=DEFAULT_WINDOW,
                eps=DEFAULT_EPS) -> jnp.ndarray:
    """One generation of mutation-selection: develop, select parents, mutate.

    Takes the selection and mutation keys **separately** rather than splitting one
    key internally. That looks fussy and is deliberate: :func:`evolve` draws them
    from a single 5-way split per generation, and any internal re-split here would
    consume the RNG stream differently and silently move every published
    canalization number. The two-key signature is what lets `evolve` delegate to
    this function while staying bit-identical to the loop it replaced.
    """
    pop_size, n = W.shape[0], W.shape[1]
    res = develop_pop(W, jnp.broadcast_to(s0, (pop_size, n)),
                      a, devsteps, window, eps)
    if stabilizing:
        fit = jax.vmap(lambda p, st: fitness(p, st, s_opt, sigma_sel))(
            res.phenotype, res.stable)
    else:
        fit = viability_fitness(res.stable)

    total = fit.sum()
    # If the population collapses (no viable individuals), reseed from
    # the founder rather than dividing by zero.
    probs = jnp.where(total > 0, fit / total, jnp.ones(pop_size) / pop_size)
    parents = jax.random.choice(k_sel, pop_size, shape=(pop_size,), p=probs)
    W_sel = W[parents]

    mut_keys = jax.random.split(k_mut, pop_size)
    return jax.vmap(lambda k, w: mutate(k, w, mask, mut_rate, mut_sd))(
        mut_keys, W_sel)


def evolve(key, founder: Network, *, pop_size=200, generations=200,
           a=DEFAULT_A, sigma_sel=1.0, mut_rate=0.02, mut_sd=0.5,
           probe_sd=1.0, tau=0.3, stabilizing=True, robustness_probes=20,
           measure_every=10, devsteps=DEFAULT_DEVSTEPS,
           window=DEFAULT_WINDOW, eps=DEFAULT_EPS):
    """Run mutation-selection from a viable founder (Siegal-Bergman protocol).

    The optimum phenotype `s_opt` is the founder's own equilibrium, so
    `stabilizing=True` applies purely stabilizing selection. `stabilizing=False`
    is the drift control (viability-only). Topology and s0 are conserved across
    the lineage; only the interaction strengths W evolve.

    RNG structure, which constrains checkpointing. The key is split SEQUENTIALLY
    down the lineage — generation `g`'s keys depend on every split before it — so
    the reproducible unit is a whole run from `key`, and a run can only be
    resumed at a generation boundary whose carry (`W` plus the current key) was
    saved. There is no way to jump to generation `g` from the seed alone, and no
    finer-than-a-generation resume without reaching inside this loop. Callers
    that checkpoint should store the carry, not just the seed.

    See the module docstring on GPU determinism before comparing two runs.

    Returns (final_population_W, EvolveHistory).
    """
    mask = founder.mask
    s0 = founder.s0
    s_opt = founder_optimum(founder, a, devsteps, window, eps)

    W = founding_population(founder, pop_size)
    hist = EvolveHistory([], [], [], [])

    def record(gen, key_r):
        m = measure_population(key_r, W, mask, s0, s_opt, a=a,
                               sigma_sel=sigma_sel, probe_sd=probe_sd, tau=tau,
                               robustness_probes=robustness_probes,
                               devsteps=devsteps, window=window, eps=eps)
        hist.generation.append(int(gen))
        hist.robustness.append(m["robustness"])
        hist.mean_fitness.append(m["mean_fitness"])
        hist.viable_fraction.append(m["viable_fraction"])

    for gen in range(generations):
        key, k_dev, k_sel, k_mut, k_rec = jax.random.split(key, 5)

        if gen % measure_every == 0:
            record(gen, k_rec)

        W = evolve_step(k_sel, k_mut, W, mask, s0, s_opt, a=a,
                        sigma_sel=sigma_sel, mut_rate=mut_rate, mut_sd=mut_sd,
                        stabilizing=stabilizing, devsteps=devsteps,
                        window=window, eps=eps)

    record(generations, key)
    return W, hist
