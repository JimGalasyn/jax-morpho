"""The developmental `RunFn`s behind the campaign contract (DESIGN.md §5b).

Proof-of-use for `run_farm` (the extracted campaign layer). This is the ONE place
morpho physics meets the boundary: it imports the evodevo engine and exposes a
single callable matching `run_farm.RunFn`. Nothing in run-farm imports anything
here.

`evodevo_run` dispatches on ``config.kind`` so a whole campaign — every arm — rides
one RunFn (`run_campaign` takes exactly one):

  * ``"evolve"`` — one lineage (§5b's run unit): a population developed, selected,
    and reproduced for ``n_generations``. The loop is driven **one generation at a
    time** so it can stream a per-generation ledger row (P6) and checkpoint the full
    genotype state (P4/contract B) between generations. Each generation draws its
    RNG from a seed derived deterministically from ``(seed, gen)``, so a preempted
    lineage that resumes at generation *g* replays a bit-identical stream — restart
    is exact, not merely restart-*able*.
  * ``"mu_gate"`` — the Milocco–Uller Fig 3C known-answer gate (``reference_mu``).
    Making the calibration an arm any campaign can include means **every campaign
    carries its own validation ladder** (DESIGN.md §5b): the check re-runs alongside
    the result instead of being cited from our test suite. It emits the G-vs-P
    angle per allele-frequency point — G at the noise floor, P tens of degrees off,
    is the signature that the developmental G is real.

Development can create neighbour-exchange basin jumps (the multistability of §2E),
so a lineage that fails to equilibrate is a RECORDED fact in the result, never a
silent NaN — the same relax-then-read honesty the soliton RunFn keeps.
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from run_farm.protocols import RunContext

from jax_morpho.evodevo import evolution as EV
from jax_morpho.evodevo import genetics as GEN
from jax_morpho.evodevo import genome_map as GM
from jax_morpho.evodevo import phenotype as PH
from jax_morpho.evodevo import pipeline as PL
from jax_morpho.evodevo import reference_mu as MU
from jax_morpho.evodevo import response as RS
from jax_morpho.runs import RunConfig


def _gen_seed(base_seed: int, gen: int) -> int:
    """A per-generation RNG seed derived deterministically from ``(base_seed, gen)``.

    `evolve` reseeds a fresh Generator from a single int each call, so driving it
    one generation at a time needs a distinct-but-reproducible seed per generation.
    A hash (not ``base_seed + gen``) keeps neighbouring lineages' streams from
    overlapping while staying a pure function of the pair — the property restart-
    exactness rests on: a run resumed at generation *g* re-derives the identical
    seed for every remaining generation.
    """
    h = hashlib.sha256(f"{base_seed}:{gen}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def _build_arena(config: RunConfig):
    """Rebuild the fixed developmental arena (organism, architecture, optimum)
    from the config alone — deterministic, so a resumed worker reconstructs the
    identical arena the original ran in. Returns ``(org, arch, opt, n_env)`` with
    ``org``/``opt`` = None for a neutral (develop=False) lineage.
    """
    p = config.params
    n_env = int(p.get("n_env", 2))
    arch = GEN.make_architecture(
        config.n_genes, int(p.get("n_loci_per_gene", 5)),
        float(p.get("sigma_gamma", 0.04)),
        np.random.default_rng(int(p.get("arch_seed", 0))))

    if not p.get("develop", True):                 # neutral-drift lineage
        return None, arch, None, n_env

    grn = GM.init_grn(
        jax.random.key(int(p.get("grn_seed", 0))), config.n_genes + n_env,
        hidden=int(p.get("grn_hidden", 16)), scale=float(p.get("grn_scale", 1.5)))
    org = PL.make_organism(
        grn, jnp.zeros(config.n_genes + n_env),
        n_rings=int(p.get("n_rings", 2)),
        landmark_stride=int(p.get("landmark_stride", 5)))
    opt = RS.reference_optimum(
        org, arch, sigma_env=float(p.get("sigma_env", 0.005)),
        optimum_scale=float(p.get("optimum_scale", 4.0)))
    return org, arch, opt, n_env


def _make_selection(config: RunConfig, opt):
    """The selection seam, from ``params['selection']``."""
    s = dict(config.params.get("selection", {"type": "truncation", "frac": 0.2}))
    if s["type"] == "neutral":
        return EV.neutral_selection(float(s.get("frac", 1.0))), False
    if s["type"] == "truncation":
        if opt is None:
            raise ValueError("truncation selection needs a developed arena "
                             "(params['develop'] must not be False)")
        return EV.truncation_toward(opt, float(s.get("frac", 0.2))), True
    raise ValueError(f"unknown selection type: {s['type']!r}")


def _make_variation(config: RunConfig, arch):
    """The variation seam (§5c), from ``params['variation']``: gradualism
    (point mutation, within-basin) vs punctuation (retroviral insertion, a
    whole-gene basin jump needing a donor pool)."""
    v = dict(config.params.get("variation", {"type": "none"}))
    kind = v.get("type", "none")
    if kind == "none":
        return EV.no_variation, None
    if kind == "point":
        return EV.point_mutation(float(v.get("rate", 0.02))), None
    if kind == "retro":
        donors = np.ones((arch.n_loci, int(v.get("n_donors", 5))), dtype=int)
        return (EV.retroviral_insertion(
            float(v.get("rate", 1.0)),
            genes_per_event=int(v.get("genes_per_event", 1))), donors)
    raise ValueError(f"unknown variation type: {kind!r}")


def _row(rec: dict, gen: int) -> dict:
    """One small JSON-native event record from an evolve history row (P6)."""
    z = rec.get("z_mean")
    out = {
        "gen": int(gen),
        "n": int(rec["n"]),
        "heterozygosity": float(rec["heterozygosity"]),
        "n_selected": int(rec.get("n_selected", 0)),
    }
    if z is not None:
        z = np.asarray(z)
        out["z_mean"] = [float(v) for v in z.ravel()]
        out["z_mean_norm"] = float(np.linalg.norm(z))
    # measure_response=True adds a breeder's-prediction snapshot; carry the
    # JSON-native scalars/vectors so an aggregation can read Gβ off the ledger.
    for k in ("dz_pred", "dz_obs"):
        if rec.get(k) is not None:
            out[k] = [float(v) for v in np.asarray(rec[k]).ravel()]
    return out


def evolve_lineage(config: RunConfig, ctx: RunContext) -> dict:
    """RunFn: develop → select → reproduce for ``config.n_generations``, streaming
    a per-generation ledger and checkpointing the genotype state each generation.

    Full state = ``{"scores": (n_loci, n_pop) int genotype}`` + the generation
    counter (through ``ctx.resume_step``, not smuggled in state). A spot preemption
    resumes at the checkpointed generation with a bit-identical RNG stream.
    """
    p = config.params
    org, arch, opt, _ = _build_arena(config)
    selection, develop = _make_selection(config, opt)
    variation, donors = _make_variation(config, arch)
    measure = bool(p.get("measure_response", False)) and develop
    basis = None if not develop else PH.tangent_basis(org.ref)

    # Resume from a checkpoint (contract B) or sample a fresh population.
    if ctx.resume is not None:
        scores = np.asarray(ctx.resume["scores"])
        gen0 = ctx.resume_step if ctx.resume_step is not None else 0
    else:
        scores = GEN.sample_genotypes(
            float(p.get("p0", 0.5)), arch.n_loci, config.n_pop,
            np.random.default_rng(int(p.get("init_seed", config.seed))))
        gen0 = 0
        ctx.emit({"gen": 0, "n": int(scores.shape[1]), "event": "seeded",
                  "heterozygosity": float(EV.heterozygosity(scores))})

    last = None
    for gen in range(gen0, config.n_generations):
        # One generation = one reseeded evolve step; threading `scores` forward.
        scores, hist = EV.evolve(
            scores, org, arch, n_generations=1, selection=selection,
            variation=variation, donors=donors,
            n_env=int(p.get("n_env", 2)),
            sigma_env=float(p.get("sigma_env", 0.005)),
            develop=develop, basis=basis, measure_response=measure,
            seed=_gen_seed(config.seed, gen))
        last = _row(hist[0], gen)
        ctx.emit(last)
        # Full genotype state, so a preempted lineage resumes mid-selection.
        ctx.checkpoint({"scores": np.asarray(scores)}, gen + 1)

    het_final = float(EV.heterozygosity(scores))
    result = {
        "kind": "evolve",
        "arm": p.get("arm"),
        "replicate": p.get("replicate"),
        "n_generations": int(config.n_generations),
        "n_pop": int(config.n_pop),
        "heterozygosity_final": het_final,
        "heterozygosity_start": (last or {}).get("heterozygosity"),
        "last_gen": last,
    }
    # The rare kept full-state capture: the final population, after the lineage
    # has run to its selection limit (P7 — captured after the quench, not mid-run).
    ctx.trigger({"scores": np.asarray(scores),
                 "het_final": np.array(het_final)}, reason="lineage_complete")
    return result


def mu_gate(config: RunConfig, ctx: RunContext) -> dict:
    """RunFn: the Milocco–Uller Fig 3C known-answer gate (DESIGN.md §5b).

    Sweeps the θ₂ minor-allele frequency, and for each point streams the angle
    between the observed recombinant response and the G-based vs P-based breeder's
    predictions. G at the measurement noise floor while P is tens of degrees off is
    the signature that the developmental G is a real object — the check that travels
    with every campaign instead of being cited from a paper.
    """
    p = config.params
    n_ind = int(p.get("n_ind", 2000))
    n_replays = int(p.get("n_replays", 20))
    p2_values = p.get("p2_values")
    if p2_values is None:
        p2_values = list(0.5 / 2.0 ** np.arange(9, -1, -1))

    worst_angle_G = 0.0
    rows = []
    for k, p2 in enumerate(p2_values):
        r = MU.simulate_fig3c(float(p2), n_ind=n_ind, n_replays=n_replays,
                              seed=config.seed + k)
        row = {
            "p2": float(r["p2"]),
            "angle_G": float(r["angle_G"]),
            "angle_P": float(r["angle_P"]),
            "angle_G_sens": float(r["angle_G_sens"]),
            "G_rel_frob": float(r["G_rel_frob"]),
        }
        rows.append(row)
        worst_angle_G = max(worst_angle_G, row["angle_G"])
        ctx.emit(row)

    median_angle_P = float(np.median([r["angle_P"] for r in rows]))
    return {
        "kind": "mu_gate",
        "arm": p.get("arm", "mu_reference"),
        "n_points": len(rows),
        "worst_angle_G": worst_angle_G,
        "median_angle_P": median_angle_P,
        # the gate: G tracks (small angle), P does not (large angle). A campaign
        # consumer asserts on these to know its engine still reproduces Fig 3C.
        "gate_pass": bool(worst_angle_G < 20.0 and median_angle_P > worst_angle_G),
    }


def evodevo_run(config: RunConfig, ctx: RunContext) -> dict:
    """The single injected RunFn for a morpho campaign; dispatches on ``kind``."""
    if config.kind == "evolve":
        return evolve_lineage(config, ctx)
    if config.kind == "mu_gate":
        return mu_gate(config, ctx)
    raise ValueError(f"unknown run kind: {config.kind!r} "
                     "(expected 'evolve' or 'mu_gate')")
