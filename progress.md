# progress.md — SmoreFit.jl Session Journal

> **Purpose:** Session-level decisions, rejected approaches, and open questions.
> Unlike [PRD.md](PRD.md) (specification) and [README.md](README.md) (completion status), this file captures the *reasoning* behind decisions — things that would otherwise exist only in ended chat history.

---

## Session: Initialization — Architecture decisions relevant to SmoreFit (2026-05-19)

### Key Decisions

**SmoreFit is where real-world data enters the pipeline**
SmoreBase fits the SM to CM-generated training data and quantifies SM parameter uncertainty. SmoreFit is the next step: given real-world observational data, use the SM (within its uncertainty region) as a fast likelihood evaluator to infer a posterior over CM parameter space.

**Planned API: `buildPosterior`**
`buildPosterior(sm, realWorldData, uqResults, cmBounds; ...) -> CMPosteriorResult`. Exact signatures and types are TBD — see [PRD.md](PRD.md) for open questions.

**`ParameterPrior` (from SmoreBase) is the natural type for CM bounds**
The same `ParameterPrior` type used for SM parameter priors can hold CM parameter priors, since it supports full `Distributions.jl` distributions. This makes SmoreFit a natural consumer of SmoreBase types without duplicating the bounds concept.

### Status
Nothing implemented. Package is a stub. Implementation deferred until the SmoreBase pipeline is validated end-to-end in SmoreExamples.

---

## Session: `buildPosterior` implementation + CM-sample unification (2026-05-22)

Implemented the core `buildPosterior` feature (Phase C) on top of a cross-package refactor
(Phases A–B). Adapted from the MATLAB SMoRe ParS `acceptSampledABMParameters.m`, deliberately
**not** ported wholesale.

### The bridge
Both the CM (per cohort point) and the real data, through the shared SM, induce confidence
regions in the **same SM-parameter space**. A CM parameter set is in the posterior iff its SM
region overlaps the data's. SmoreFit profiles the SM against the data (reusing SmoreBase's
`fitSurrogate` + `_uq`), then scores each cohort point's overlap with the data region.

### Key decisions
- **Three selectable bridge methods** (`bridge` kwarg), replacing MATLAB's six-way
  `acceptance_method` sprawl: `:box_overlap` (symmetric box-vs-box), `:data_trace_in_box`
  (data profile trace points inside the CM box, = MATLAB `all_profiles` de-sprawled),
  `:symmetric_trace`.
- **Marginal χ²(1) CIs**, not MATLAB's χ²(k)-on-1-D. A joint χ²(k) cutoff applied to a profile
  where only one parameter was held fixed is not a valid confidence statement. We reuse
  SmoreBase's `ProfileCurve.ci_lower/ci_upper` directly. (We do **not** recompute thresholds.)
- **`nothing` CI bound → swept-range extreme** (the SM prior/sweep edge), not ±Inf, so the
  prior still bounds an unidentified direction.
- **Real data = single-param-set `CMData`** — no new type (resolves the PRD "type TBD").
- **Row-aligned `uq_results` + `cm_sample`**, mirroring SmoreGSA's `runSensitivity`. A
  `ProfileLikelihoodResult` does not record its CM point, and SmoreBase cannot attach it
  (`CMData` holds only summary stats, never CM input params), so the pairing is the caller's
  by row order — no bundle struct.
- **Grid-aware result.** `CMPosteriorResult` stores the `AbstractCMSample`, so `acceptedGrid` /
  `scoreGrid` reshape onto the CM grid. This is what the next step (interior interpolation)
  needs.
- **Posterior kwarg**: `:accept` (hard set via `posteriorSamples`) or `:graded` (normalized
  scores via `posteriorWeights`). Gradation is intentionally simple for now.

### CM-sample unification (Phases A–B)
Lifted `AbstractCMSample` / `GridCMSample` / `ScatteredCMSample` from SmoreGSA into SmoreBase
(`src/types/cm_sample.jl`), added a `CMSample(matrix)` factory and a `reshapeToGrid` helper
(de-duplicating logic that was inlined in SmoreGSA). SmoreGSA now consumes the SmoreBase types
(its test suite stays green). SmoreBase bumped to 0.1.3. Done now (rather than later) because
the next step — interpolating the posterior onto interior CM points — needs the grid structure,
and both siblings should share one abstraction. Local SmoreBase is `dev`-linked into SmoreGSA
and SmoreFit so the unified types resolve.

### Scoring caveat (recorded for later)
The `:data_trace_in_box` / `:symmetric_trace` *fraction* of trace points inside a box is
sampling/reparameterization-dependent (we sample evenly in the profiled parameter, not arc
length). The boolean accept ("any point inside") is invariant; only the graded fraction is the
proxy. The principled replacement is an arc-length / Jacobian-weighted integral. Also dreamed:
a trace-distance "street area" measure between the two profile traces.
