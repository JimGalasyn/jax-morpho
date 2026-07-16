# jax-morpho.evodevo — the tiered environment & ecology system

Status: **proposal, for review.** Nothing here is built. This document exists to
fix the architecture before any of it is written, in the same spirit as
`DESIGN.md` — and to be argued with first.

Companion to `docs/DESIGN.md`. That document owns the spine
`genotype → development → phenotype → selection → response` and its layers A–E.
This one owns the **environment**: what it is, where it enters that spine, which
parts of it may touch a gradient, and what has to be true before any of it earns
a number.

## 0. Thesis

> **The environment is not a physics problem here. It is a fitness problem.**

Selection today is `quantgen.truncation_select(z, optimum, keep_fraction=0.5)` —
keep the half of the population closest to a **fixed optimum shape we supply**,
in Procrustes tangent coordinates. `response.py:88-97` documents that the optimum
*must* stay fixed across the sweep, and §3d records why: recomputing it per
population was one of the two protocol errors that produced a plausible-looking
null.

That constraint is correct **for Phase 0–3, and it is exactly what makes them
gates.** A fixed, known optimum is what lets gate #3 ask "does `Δz̄ = Gβ` predict
the response?" and get a checkable answer. Calibration needs an exogenous target.

But it means the loop *cannot respond to an environmental stressor*, in the
strict sense: the thing selection points at is a constant we chose. Bolt the most
faithful thermal/chemical/mechanical field stack imaginable onto development and
the target still does not move. The environment would perturb development
(plasticity) while selection kept pulling toward the same hand-picked shape. That
is plasticity theatre, and no amount of environment *physics* fixes it.

So the proposal is not primarily "add fields." It is:

> **Make fitness a functional of form, computed from the environment — and get
> `β` from physics instead of from a hand-chosen optimum.**

`Δz̄ = Gβ` has two factors. `G` is derived (gate #2, `1.85e-03`; gate #3, error at
the noise floor). `β` is currently declared. Deriving `β` the same way — from
uptake through the developed body's boundary — would make **both sides of Lande's
equation derived from the mechanical engine, with no free parameters on either.**
That is a stronger claim than Phase 2 or Phase 3, it is forward/forbidding rather
than a fit, and it is the reason to do this work now rather than as a game
feature later.

**This stays a sufficiency/construction contribution** (§0 of DESIGN.md). "An
engine in which selection gradients follow from physics" — never "this is how
selection works."

## 1. The organising principle: differentiability is bought with timescale separation

The tiers are not a taxonomy of environmental phenomena. They are a **ladder of
timescale ratios**, and the ratio is what decides whether a tier may touch a
gradient.

The relevant clock hierarchy:

    τ_mech  ≪  τ_field  ≪  τ_life  ≪  τ_gen  ≪  τ_evo
    relax       field       uptake,    one       many
    to x*       reaches     encounter  generation generations
                steady

Read left to right, each `≪` buys something specific:

- **`τ_mech ≪ τ_field`** — the form is at equilibrium before the field responds,
  so **the field is a function of the form**, `φ = φ(x*)`. If this fails, field
  and form are coupled dynamics and there is no fixed point to differentiate.
- **`τ_field ≪ τ_life`** — the field is at steady state while fitness accrues, so
  **fitness is a functional of `(x*, φ)`**, not of a trajectory. If this fails,
  fitness requires integrating a PDE over a lifetime and the gradient cost
  explodes.
- **`τ_life ≪ τ_gen`** — an individual's ecology resolves within a generation, so
  the ecology tier can be a **separate, coarse simulation** that hands a scalar
  back. If this fails, development and ecology interleave and the tiers collapse.

Where a separation holds, the tier is a **steady-state solve** — and a
steady-state solve is either linear (trivially differentiable) or a fixed point
(differentiable by the same IFT machinery `fixed_point.py` already implements, at
one linear solve per VJP, §1 of DESIGN.md). Where a separation fails, the tier is
genuinely dynamic and **must be kept out of the gradient path**.

That is the whole design. Everything below follows from it.

**Locked (proposed):** a tier is admitted to the gradient path **only** by
exhibiting its separation, with a measured ratio. No tier gets differentiated
because it would be convenient.

## 2. The tiers

| tier | what | timescale | in gradient path? | enters spine at |
|---|---|---|---|---|
| **0** | quasi-static fields `φ` — thermal, chemical-from-fixed-sources, mechanical load | `τ_field` | **yes** | layer A (sensing) + layer B (loading) |
| **1** | endogenous fitness `W` — flux integrals over `φ` at `x*` | `τ_life` | **yes** | layer D (`β = ∂W/∂z`) |
| **2** | ecology — organism-as-particle, traits `= z`, shared resource pool | `τ_life`–`τ_gen` | **no**, and does not need to be | layer E (selection) |
| **3** | non-stationary chemistry — Gray-Scott in its interesting regimes | `~τ_life`, or never | **no**, and *cannot* be | layer A as stochastic input |
| **∞** | planetary boundary condition — exoclimate | `≫ τ_gen` | no | sets tier 0's far-field value |

### Tier 0 — quasi-static fields

**`φ` is a per-cell field, exactly like `θ`.** That is the entire trick.

`equilibrate(pos, alive, theta, ...)` does not care where a parameter came from.
Add `φ` alongside `θ` and `fixed_point.fixed_point_sensitivity` returns
**`∂x*/∂φ` from the identical solve**. Plasticity gradients are free — not "cheap
to add," *already implemented*, applied to a new argument.

Three channels, all solved on a graph Laplacian over the contact topology, all
with Dirichlet boundaries:

- **chemical `c`** — `∇·(K∇c) = 0` with sources/sinks; `K` from the local `θ`
  field (permeability as a developmental parameter).
- **thermal `T`** — same operator, conductance instead of permeability, far-field
  value from tier ∞.
- **mechanical load `σ_ext`** — an external potential term; the only one that
  enters the *force law* rather than only the readout.

`Morphospace/morphospace/physics/k7_transport.py`'s `MorphogenField` — explicit
`source_cells`/`sink_cells` with values and `apply_boundary_conditions()` — is
the right abstraction and should be copied in shape. It is **numpy with
string-keyed cell identity (`cell_aas`) load-bearing all the way down**, so it is
**not** incrementally portable. **Locked (proposed): rebuild tier 0 in JAX in
`jax_morpho.evodevo.field`; steal the design, port no code.** It is a Laplacian
assembly plus a linear solve — genuinely small.

### Tier 0 enters the spine twice, and they are different

1. **Sensing → layer A.** `grn_field(genome, coords, grn)` computes
   `θ_i = MLP([a, u_i])` where `u_i` is cell `i`'s positional coordinate. `coords`
   is documented as "positional information read by the GRN" — it *is* the sensing
   seam. Sensing is: **append `[c_i, T_i, σ_i]` as extra columns of `coords`.**
   Nothing else changes. `n_in` goes from `n_genes + 2` to `n_genes + 5`.

   This is the same idea as Morphospace's `MECHANO` pseudo-TF (stress written into
   a motif's concentration slot, so any gene with a `GATAC` site responds through
   the ordinary sigmoid pool). That design is good and should be credited; in
   jax-morpho's MLP formulation it is just more input columns.

2. **Loading → layer B.** `field_morse_energy` is purely pairwise: no ambient
   term, no substrate, no boundary. Loading is `E_total = E_pair + Σ_i V(x_i; φ)`.

   Precedent exists: `inverse_design.confinement_energy` is a real external
   potential `V = ½(kx·x² + ky·y²)` with gradients reaching the field parameters.
   But it sits on the **legacy `center_based` scalar path with the non-converging
   clipped relaxation** (§1) and is not wired to `mechanical.py` at all. Take the
   idea; leave the code.

**Sensing without loading is the cheap half and is not enough.** A GRN that reads
temperature but whose mechanics ignore it gives an organism that *knows* about
the environment and is not *in* it. Both, or neither.

### Tier 1 — endogenous fitness

`W = ∮ K ∇c · n̂ dA` over the body boundary — uptake through the surface, computed
on the tier-0 field around the developed form.

Why this is the right shape:

- It is a **linear solve → trivially differentiable**, so `β = ∂W/∂z` comes out of
  autodiff composed with the existing implicit path. The whole `lande_response_vjp`
  two-solve machinery survives untouched.
- It makes **form selectable because of what form does**: surface-to-volume,
  gradient exposure, self-shadowing. Nobody chose it.
- It is **density- and frequency-dependent for free** once tier 2 puts more than
  one organism in the pool — organisms deplete a shared field, so a genotype's
  fitness depends on who else is present. That is the precondition for ecology,
  and it arrives as a consequence rather than a feature.
- `phenotype.centroid_size(L)` already exists, so the size axis that trophic
  interaction needs is already in the readout.

**Not claimed:** that uptake is the right fitness for a real organism. It is *a*
fitness that is a functional of form rather than a target we picked. The claim is
structural, not biological.

### Tier 2 — ecology

**Organisms are particles. Traits are `z`. `z` is already the interface.**

You do not simulate two 10⁶-cell bodies interacting mechanically, and you should
never want to. Development is per-organism and fine-grained; ecology is coarse
and reads the developed phenotype. `phenotype()` produces the Procrustes shape
vector; ecology consumes it as a trait vector. **This layer requires no new
representation.**

The tier-2 state is organism position, the shared resource field, and a trait
vector per individual. Interactions: uptake (deplete), death (emit biomass back
into the pool), encounter (see §5).

**This tier is not differentiable and does not need to be.** Fitness reaches the
genome through tier 1, which *is* differentiable. Tier 2 only decides *how much*
of tier 1's field each organism gets. Gradients do not have to cross it.

### Tier 3 — non-stationary chemistry, and the honest limit

`reaction_diffusion.py`'s tuned Voronoi-robust Gray-Scott regimes are a genuine
asset (5 Pearson regimes: moving_spots, negatons, spirals, stripes, stable_spots).
And the interesting ones — moving spots, spirals — are interesting **precisely
because they have no steady state.** `τ_field ≪ τ_life` fails by construction.

**Locked (proposed): tier 3 is permanently outside the gradient path.** Not "not
yet" — the separation cannot be recovered, because its absence is the phenomenon.
Tier 3 is a **stochastic input to layer A**: it writes into `coords` like tier 0
does, and `∂z/∂a` is taken at fixed tier-3 realisation. Averaged over
realisations, this is a valid sensitivity of the *expected* phenotype. Attempting
to backprop through 5000 Euler steps of Gray-Scott is the `center_based.relax`
mistake with extra steps: differentiating a trajectory that is not converging to
anything.

### Tier ∞ — exoclimate is the boundary condition, not a field

exoclimate is **well-mixed by explicit design**; `n_cells` is pure broadcast
(`np.full(n_cells, state.surface_T)`), documented as *"the engine has no notion of
what a cell is."* Every attempt to make it produce spatial structure will fail,
because it is 0D on purpose.

**That is not a limitation — it is the correct role.** exoclimate supplies
**tier 0's Dirichlet value at infinity**: `T_∞`, far-field chemistry. Tier 0
solves the local field with that BC. No port, no JAX conversion, no change to
exoclimate at all.

And the loop closes the other way for free, because
`AtmosphericEnvironment.step(biotic_emissions={species: delta})` is **already the
right contract** — organisms emit into a shared medium, the medium responds, the
engine knows nothing about biology. Tier 2 aggregates emissions
(`environment/metabolism.py::biotic_emissions` maps AA chemistry class → emission
dict); exoclimate steps; the far-field BC moves; tier 0 re-solves.

**The Gaia loop closes around the organism loop, and neither engine learns about
the other.** This is the first architecture in which the exoclimate stack and the
evodevo stack actually compose. Worth doing for that alone.

> **Do not reuse Daisyworld's per-cell temperature as a field.** It is the only
> per-cell environmental scalar in either repo, but the field is
> `T_planet + (1 − heat_transport)·sensitivity·(mean_α − α_i)` — an **algebraic
> deviation from the mean with no spatial coupling.** Two cells with the same
> albedo get the same T regardless of where they sit. It is not diffusion and it
> will not produce a gradient. It *is* excellent as a **gate** — see gate #8.

## 3. The consequence that bites: the null space is not a constant

**An external field breaks rigid invariance, and the Hessian's zero modes are not
a property of the code — they are a property of the total energy's symmetry.**

Today `E` is rigid-invariant, so `H` has exactly 3 zero modes in 2D (2
translations + 1 rotation; measured spectrum `[0, 0, 0, 6.36, 9.56, …]` against
`λ_max ≈ 263`). `rigid_modes(pos, alive)` returns them analytically and they are
passed as `null_basis` at `pipeline.py:147,164` and `fixed_point.py:262`, and used
inside `equilibrate`'s `newton_dir_cg`.

Add `Σ V(x_i; φ)` and this **silently becomes wrong**, in a way that depends on
the field's symmetry:

| field | breaks | zero modes remaining |
|---|---|---|
| none (today) | — | 3 |
| uniform (e.g. constant `T`) | nothing | 3 |
| linear gradient | rotation | 2 (translations survive ‖ isopleths… partially) |
| point source | translation | 1 (rotation about the source) |
| general | everything | 0 |

Projecting out `rigid_modes` when they are **no longer null** does not raise —
it **discards real physics**, quietly, and every downstream number stays
plausible. This is precisely the failure mode §3b's "Gaussian cloud is not a
tissue" was: a null-space assumption violated, symptoms indirect.

**Locked (proposed): the null basis must be *computed and gated*, never assumed.**
Replace the analytic `rigid_modes` call in the field-enabled path with a numerical
null space of `H`, and gate the declared dimension against the measured Hessian
spectrum (gate #5). This is cheap, it catches the entire class of error, and the
spectrum measurement already exists as a test pattern.

### And the gauge finding inverts: in a field, orientation becomes phenotype

This is the part I most want reviewed, because it changes layer C.

Phase 1 found the gauge is **anholonomic**: each gradient step carries zero net
torque, yet net rotation accumulates, because the modes rotate with the shape as
it deforms — a geometric phase. **The equilibrium form is a function of `θ`; its
orientation is a functional of the whole developmental trajectory.** That promoted
Procrustes from convenience to necessity: without it, "phenotype" carries a
path-dependent geometric-phase term, and `∂z/∂a` vs FD improved from ~0.7
relative to `4.84e-09` raw once the Procrustes readout annihilated the rigid modes
(`|∂z/∂x* · Z| = 1.4e-17`).

**In a field, that reasoning runs backwards.** The field pins orientation
physically. An organism aligned *with* the thermal gradient and one aligned
*across* it are **different phenotypes with different fitness** — that is the
whole point of putting it in a field. Procrustes-aligning them would quotient out
**exactly the environmental response we are trying to measure.** The tool that was
load-bearing in free space becomes an eraser in a field.

The resolution is pleasing enough that it worries me slightly, so it needs a
number rather than admiration:

> **The environment supplies the gauge that development left anholonomic.**

Layer C therefore needs a **field-relative readout**: not "quotient out all
rotation," but "express the form in the field's frame." Orientation relative to
`∇φ` is retained as phenotype; the arbitrary lab frame is still quotiented out.
As `|∇φ| → 0` the field frame becomes undefined and the readout must degenerate
continuously back to Procrustes — which is gate #6, and which is the honest way to
check that this idea is a physical statement and not a pun.

## 4. Where this sits in the sequencing — and why it is not "later"

DESIGN.md §5: Phase 4 is the evolution loop, layer E, "the closed loop / game
substrate." §2E lists its seams, and one of them is already **"selection = env or
player."**

So this is not a new layer. **It is layer E's own declared seam, and Phase 4
cannot be designed without deciding it.** If Phase 4 is built on
truncation-toward-a-fixed-optimum, we get a multi-generation loop that walks to a
shape we chose. That is *fine as gate #4* — "multi-generation loop matches
quantitative-genetic expectations" is a known-answer rung and wants an exogenous
optimum. It is inert as a science or game substrate.

**Proposed resolution: build Phase 4 with fitness as a seam, and ship both
instances.** `FixedOptimumFitness` (today's truncation, the calibration arm, keeps
gate #4 honest) and `UptakeFitness` (tier 1, the substrate). The evolution loop
never learns which it has.

This also matches §5b's standing requirement to **ship the M-U reference as a
built-in campaign arm** so every campaign carries its own known-answer gate. The
fixed-optimum fitness is the same idea one level down.

### The convergence with §5c worth naming

§5c requires **multi-lineage populations with contact between them**, because
horizontal viral transfer is meaningless in a single panmictic pool — the donor
has to come from somewhere. §5c notes this is the same substrate the campaign work
wants: "one mechanism, two motives."

**Tier 2 ecology needs the identical substrate, for a third motive.** Multiple
lineages, spatially resolved, in contact, exchanging material. Predation and
retroviral transfer are the *same graph* with different payloads. That is a real
architectural saving and an argument for designing tier 2 before Phase 4 hardens.

## 5. Predator-prey: what is reachable, and what is not

Jim's question. The honest answer is a ladder, and the top rung is out of reach —
not on compute grounds.

**Reachable, and the ecology is real.** Endogenous, density- and
frequency-dependent selection where fitness is a functional of form and organisms
compete for a shared depleting field. This is Daisyworld-class — and note that
**Daisyworld already in the tree is the existence proof**: a two-species,
frequency-dependent ecology with genuine environmental feedback, doing it through
albedo instead of predation. The mechanism is not speculative; only the payload
changes.

**Reachable with an imposed rule: size-structured trophic dynamics.** Let an
organism be a sink for other organisms, gated on a size ratio from
`centroid_size(z)`. Size-based food webs are a real, well-studied class of model,
and this would produce genuine predator-prey **cycles** — which is gate #7,
because Lotka-Volterra is a known answer.

**Be clear about what that is and is not.** The eating *rule* is imposed. This is
"trophic interaction mediated by evolved morphology," not "predation emerged from
morphology." The distinction is exactly DESIGN.md §0's no-menu-fit rule, applied
to ecology: a demo where predation appears because we wrote a predation rule is a
**sufficiency demo**, and must be labelled one.

**Not reachable, and the reason is structural, not computational.** Behavioural
predation — pursuit, evasion, chemotaxis toward prey — requires the organism to
*act*, and the loop has no acting in it. Development runs to equilibrium and we
measure the form. There is no time-in-life, no sensing-then-moving, no controller.
Spore's cell stage develops a body **and then the body swims and eats**; we have
the first half. Adding the second half means a controller and a life-clock. That
is a larger project than this entire document and should not be smuggled in as a
tier.

> **On scale, gently: 10⁶ cells is not a microbe.** A microbe is 1 cell.
> *C. elegans* is 959. Hydra is ~10⁵. 10⁶ puts you at roughly a small planarian —
> and flatworms *do* have predator-prey, precisely because they have tissues,
> sensing, and a nervous system. The instinct that predation is reachable at that
> scale is right. But "microbe" smuggles in an intuition (simple, reactive, no
> controller) that the cell count contradicts. **If the target is the Spore cell
> stage, that is ~10²–10³ cells and a chemotaxis controller — and it is a *different
> and much cheaper* project than 10⁶ cells.** Worth deciding which one is wanted
> before either is built.

## 6. Scale: the tiers are also a resolution ladder

**The evodevo loop runs at 19 cells.** `n_rings=2` in every test and demo —
`test_phenotype.py:31`, `test_quantgen.py:39`, `test_response.py:59`,
`demo_phase1_gate.py:47`, `demo_phase2_gate.py:36`. The 10³–10⁶ numbers all live
on the `center_based`/`scale` path, which is the **legacy scalar-parameter engine
that does not take a `θ` field and does not converge** (§1). Two engines; only one
is the science.

That is less bad than it sounds, because **`grn_field` is already a continuum
field sampled at coords**: `θ_i = MLP([a, u_i])`. The genome→θ map is
**resolution-independent by construction** — the same genome evaluates at any cell
count. Develop coarse for gradients, render fine for the game. This is DESIGN.md
§5b lever 4 ("resolution ladder… unmeasured; worth a gate of its own") and tier 0
should be built to respect it: **fields are solved at the development resolution,
whatever that is.**

The wall is elsewhere and is unaffected by any of this: `field_morse_energy` is
dense O(N²), and §2E measured that the *binding* constraint before that is the
**descent stage** (at N=1261 CG-Newton exhausts 5000 iterations at `max|F| ~ 1`).
Organism scale needs FIRE/Nesterov globalisation, then neighbour lists. **Tier 0
neither helps nor hurts that** — the Laplacian solve is on the same contact graph
the energy already needs — but tier 0 should not be blamed for it either.

## 7. Gates

House rule (§4 of DESIGN.md): **no layer ships without its number.** Extending the
validation ladder; rungs 1–3 are passed, rung 4 is Phase 4's.

- **Gate #5 — the null basis is measured, not assumed.** For each field symmetry
  class (none / uniform / gradient / point source), the number of numerically-zero
  Hessian eigenvalues equals the declared null-basis dimension, with a clear
  spectral gap. *Target: zero modes at machine zero, first nonzero ≥ O(1) against
  `λ_max`, as in the existing `[0, 0, 0, 6.36, …]` measurement.* This is the gate
  that catches §3's whole error class.
- **Gate #6 — the field-frame readout degenerates to Procrustes.** As
  `|∇φ| → 0`, the field-relative phenotype → the Procrustes phenotype, and
  `G(φ) → G(0)`. *Target: relative difference shrinking with `|∇φ|`, controlled by
  it — the same shape as gate #2's σ-controlled claim, and stated the same way.*
  This is what makes §3's "the environment supplies the gauge" a physical claim
  rather than a slogan.
- **Gate #7 — `∂x*/∂φ` ⟺ finite differences.** Gate #1 for the environment, and
  it should be nearly free: same machinery, new argument. *Target: `≤ 1e-9` on the
  gauge-appropriate subspace, matching gate #1's `8.44e-10`.* **If this is not
  nearly free, the tier-0 design is wrong** — that is the point of routing `φ`
  through the same solve as `θ`, and this gate is the test of the whole premise.
- **Gate #8 — the reaction norm exists and is not noise.** One genome, two
  environments, two forms, distinguishable above the phenotypic noise floor —
  stated against the measurement's own resolution the way gate #3 was, not against
  an arbitrary threshold.
- **Gate #9 — derived `β` reproduces the response.** The Phase-4 payoff: with
  `UptakeFitness`, `Δz̄ = Gβ` with **both factors derived** predicts the observed
  recombinant response, at the noise floor. *This is the result the document is
  for.*
- **Gate #10 — the ecology reproduces Lotka-Volterra.** With morphology frozen,
  tier 2 recovers classical predator-prey cycles. *Known answer, cheap, and the
  correct discipline: gate the ecology against classical theory exactly as `G` is
  gated against M-U.*
- **Gate #11 — the feedback loop reproduces Daisyworld.** Tier 2 → emissions →
  exoclimate → far-field BC → tier 0 recovers temperature regulation across a
  luminosity ramp. *Known answer, the reference implementation is already in
  Morphospace, and it gates the tier-∞ coupling end to end.*

## 8. Proposed sequencing

1. **Tier 0, thermal only, uniform → gradient.** The smallest thing that exercises
   the whole seam: one channel, one Laplacian, both entry points (sensing +
   loading). Gates #5, #7. Chosen first because thermal has the simplest boundary
   condition and the null-space consequence appears immediately.
2. **The layer-C field-frame readout.** Gate #6. Must land before any multi-channel
   work, because every later number depends on the readout being right.
3. **Tier 0 chemical + mechanical load.** Gate #8 — the reaction norm. This is the
   first *result* rather than an instrument.
4. **Tier 1 + the Phase-4 fitness seam.** `FixedOptimumFitness` /
   `UptakeFitness`. Gate #9. **The payoff.**
5. **Tier 2 ecology.** Gates #10, #11. Shares the multi-lineage substrate §5c
   already requires.
6. **Deferred, and named so nothing precludes them:** tier 3 as stochastic input;
   the life-clock and controller that behavioural predation would need; the
   resolution-ladder gate (§5b lever 4).

## 9. Guardrails / non-goals / what is not claimed

- **Every tier exhibits its timescale separation with a measured ratio, or it does
  not enter the gradient path.** No exceptions for convenience.
- **Tier 3 is permanently out of the gradient path**, not deferred. Its
  non-stationarity is the phenomenon, not a limitation.
- **Sufficiency, never explanation** (§0, §6 of DESIGN.md). "An engine in which
  selection gradients follow from physics." Not "this is how selection works."
- **Predation-from-a-rule is a sufficiency demo and must be labelled one.** See
  §5. This is the menu-fit rule applied to ecology, and it is the most likely place
  for this project to fool itself.
- **Uptake is not claimed to be the right fitness for a real organism.** It is a
  fitness that is a functional of form rather than a target we chose. The claim is
  structural.
- **Do not port Morphospace's environment stack.** numpy, Python loops over
  cells×genes×sites, string-keyed cell identity load-bearing to the bottom. Steal
  the designs (`MorphogenField`'s Dirichlet abstraction; `MECHANO`-as-pseudo-TF);
  rebuild small in JAX.
- **Do not make exoclimate spatial.** It is 0D on purpose and the correct role is
  the far-field boundary condition.
- **Do not warm-start development from a reference equilibrium** to pay for tier
  0's cost — §2E already flags this as a cheat that biases which basin development
  lands in and would suppress the §3c multistability.
- **Adaptive fidelity must not skip developing organisms in novel environments**,
  for the same reason §5b lever 2 forbids skipping viral macromutations: a `G`-based
  surrogate would make the environmental response self-confirming and destroy the
  ability to observe where the linearisation fails. **Multi-fidelity, never
  surrogate-replacement.**

## 10. Open questions for review

1. **Is the fitness reframe (§0) right?** It is the load-bearing claim. If
   endogenous fitness is *not* the precondition for environmental response, the
   sequencing changes completely and tiers 0/3 become ordinary features.
2. **Does the gauge inversion (§3) hold?** "The environment supplies the gauge that
   development left anholonomic" is either a genuine physical statement or a pun
   that survived because it is pretty. Gate #6 is designed to find out, but the
   argument should be attacked first.
3. **Is `∂x*/∂φ` really free?** The premise is that `φ` is just more `θ` and the
   IFT machinery does not care. If there is a reason it *does* care — the null-space
   change is the obvious candidate — gate #7 is the wrong gate and tier 0 is the
   wrong design.
4. **Is the field-frame readout well-posed at `|∇φ| → 0`?** The field frame is
   undefined at zero gradient. Does it degenerate *continuously* or is there a
   discontinuity at zero that gate #6 would expose as a failure of the idea rather
   than of the implementation?
5. **Is tier 2 the right place for the ecology, or does it want to be a separate
   package?** It shares a substrate with §5c's multi-lineage requirement and with
   the `run-farm` campaign work — three motives, one mechanism, and possibly one
   library that is not this one.
6. **Which organism is actually wanted** — ~10²–10³ cells with a controller (the
   Spore cell stage, cheap, behaviourally interesting), or 10⁶ cells without one
   (a planarian-scale body, expensive, morphologically interesting)? §5's ladder
   says these are different projects. The rest of the plan is unaffected; the
   compute budget is not.

## 11. Prior art / credit

Milocco & Uller (2026 PNAS) own the sensitivity-derived quant-gen concept and the
environmental input `u` (§3, §3d — "Environment is load-bearing"). Their `u` is a
**non-heritable scalar input to the developmental map**, not a spatial field: it
exists so `P ≠ G`. Ours (`genetics.sample_environment`, `response.py:56`) is the
same, and tier 0 does **not** replace it — it is a different object, and both
should coexist. **`G(φ)` — how `G` itself changes with environment — is a live
question in that literature and is the natural scientific target once tier 0 and
gate #8 exist.**

Daisyworld (Watson & Lovelock) for the environmental-feedback gate. Lotka-Volterra
for the ecology gate. `MECHANO`-as-pseudo-TF is Morphospace's own idea (a collapsed
YAP/TAZ cascade) and is good; it should be credited when it reappears as `coords`
columns.
