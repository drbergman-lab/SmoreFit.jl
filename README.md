# SmoreFit

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://drbergman-lab.github.io/SmoreFit.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://drbergman-lab.github.io/SmoreFit.jl/dev/)
[![Build Status](https://github.com/drbergman-lab/SmoreFit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/drbergman-lab/SmoreFit.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/drbergman-lab/SmoreFit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/drbergman-lab/SmoreFit.jl)

Posterior inference on complex model (CM) parameter space for the [Smore](https://github.com/drbergman-lab/Smore.jl) surrogate modeling ecosystem. SmoreFit takes a fitted surrogate model (SM) from [SmoreBase.jl](https://github.com/drbergman-lab/SmoreBase.jl) and real-world observational data, and builds a posterior over CM parameters.

---

## How it works

The Smore pipeline produces, for each CM cohort parameter set, a **profile likelihood** of the surrogate model's parameters fit to the CM-generated data. SmoreFit then profiles the *same* surrogate against the **real observational data**, putting both in the same SM-parameter space, and bridges the two — a cohort point is in the posterior iff its SM-parameter confidence region overlaps the data's.

## Usage

```julia
using SmoreBase, SmoreFit

# Inputs you already have from the upstream pipeline:
#   sm         :: AbstractSurrogateModel             — the SM used for cohort profiling
#   data       :: CMData (n_param_sets == 1)         — real observational data
#   uq_results :: Vector{ProfileLikelihoodResult}    — one per cohort point (from quantifyUncertainty)
#   cm_params  :: Matrix [n_cohort × n_cm_params]    — row-aligned with uq_results
#   cm_prior   :: ParameterPrior                     — CM parameter names + support

post = buildPosterior(sm, data, uq_results, cm_params, cm_prior)

posteriorSamples(post)   # accepted CM parameter vectors (the posterior as a sample set)
posteriorWeights(post)   # scores normalized to a discrete graded posterior
acceptedGrid(post)       # accepted reshaped onto the CM grid (GridCMSample only)
scoreGrid(post)          # scores reshaped onto the CM grid
```

### If you already hold an `SMFitProblem`

`buildPosterior` also dispatches on `SMFitProblem` directly — convenient when the SM, real data, SM prior, and loss are already bundled:

```julia
problem = SMFitProblem(sm, data, sm_prior)
post    = buildPosterior(problem, uq_results, cm_params, cm_prior)
```

The `problem` form uses `problem.loss`; the `(sm, data, …)` form takes a `loss` kwarg (default `GaussianNLL()`).

### Bridge methods

`bridge::Symbol` selects how the two SM-parameter confidence regions are compared:

| Bridge | Score |
|---|---|
| `:box_overlap` *(default)* | Relative overlap volume of the two hyper-rectangles (per-parameter marginal χ²(1) CIs). Symmetric. |
| `:data_trace_in_box` | Fraction of data profile trace points (in-CI portion) that lie inside the CM box. |
| `:symmetric_trace` | Max of the two-way trace-in-box fractions. |

A `nothing` CI bound (unidentified on that side) falls back to the profile's swept-range extreme — *not* ±Inf — so the prior still bounds the region.

### Posterior representation

`posterior::Symbol`:

- `:accept` *(default)* — hard accept/reject via `score > acceptance_tol`. `posteriorSamples` gives the accepted CM vectors.
- `:graded` — keep the scores as a continuous graded posterior. `posteriorWeights` normalizes them.

### Interior CM-point queries

The result stores the cohort `uq_results` and a prebuilt CI-bound interpolator over the CM grid, so you can query any interior CM point without re-running anything:

```julia
posteriorScore(post, [θ1, θ2])              # score ∈ [0,1] at one interior point
inPosterior(post, [θ1, θ2])                 # Bool, using post.acceptance_tol
inPosterior(post, queries; tol = 0.05)      # batch + tol override; queries is [N × n_cm_params]
```

Interior queries require a `GridCMSample` layout and `bridge ∈ (:box_overlap, :data_trace_in_box)`; `:symmetric_trace` and scattered layouts raise a clear `ArgumentError`. The interpolator is selected via the `interp` kwarg on `buildPosterior` (default `LinearCIInterp()`).

> **Note on cohort sampling density.** Linear bound interpolation is reliable when consecutive
> cohort CIs overlap each other *in SM-parameter space* — equivalently, when the bound surface
> moves only a fraction of its own width between adjacent cohort points. The relevant
> dimensionless ratio compares SM-space quantities top and bottom: how far the bound shifts
> (an SM-parameter quantity, with the CM step folded into the finite-difference numerator)
> versus how wide the CI is (also an SM-parameter quantity). Three different things make that
> ratio large — a steep manifold, tight CIs, or a coarse CM cohort — and the interpolator
> can't distinguish them; in each case the interpolated box jumps past the data box, and
> interior queries can return zero even near a cohort point that *is* in the posterior. The
> only knob the user controls is cohort density, so the corrective action is the same in all
> three: add cohort points where the geometry is changing fast. A refinement diagnostic that
> surfaces this is a planned next step.

## Background

- [PRD.md](PRD.md) — behavioral spec and acceptance criteria.
- [progress.md](progress.md) — session-level decisions and the reasoning behind them.

---

## Implementation Status

> For Claude Code sessions: this section is the authoritative record of what has been built. Update it as features are completed. See [PRD.md](PRD.md) for behavioral specifications and [progress.md](progress.md) for decision rationale.

### Complete

- [x] `buildPosterior` — posterior on CM parameter space given real-world data + SM UQ from
  SmoreBase. Profiles the SM against the data, then bridges those profiles against the
  per-cohort CM profiles via one of three selectable methods (`:box_overlap`,
  `:data_trace_in_box`, `:symmetric_trace`). Returns a grid-aware `CMPosteriorResult`
  (`posteriorSamples`, `posteriorWeights`, `acceptedGrid`, `scoreGrid`). Real data is a
  single-param-set `CMData`; CM parameter locations use the shared `AbstractCMSample` types.
  Dispatches on either `(sm, data, …)` or an `SMFitProblem` already in hand.
- [x] Interior CM-point queries — `inPosterior(post, θ_cm)` and `posteriorScore(post, θ_cm)`
  (with batch + `tol` override). Interpolates the per-cohort SM-parameter CI bounds across
  the CM grid via `LinearCIInterp` and evaluates the chosen bridge against the data profile.
  `GridCMSample` + `:box_overlap` / `:data_trace_in_box` only; other configurations error.
