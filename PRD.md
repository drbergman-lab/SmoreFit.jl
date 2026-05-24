# Product Requirements Document — SmoreFit.jl

> **Purpose:** This document defines the complete feature set of SmoreFit in behavioral terms. It is the authoritative answer to "what should this system do?" Read this at the start of any feature session to establish alignment between intent and implementation plan.

---

## Product Overview

**Vision:** SmoreFit is where real-world observational data enters the Smore pipeline. Given a fitted surrogate model (SM) and its uncertainty quantification (from SmoreBase), SmoreFit infers a posterior distribution over complex model (CM) parameter space by comparing SM predictions to real-world observations.

**Target Users:** Computational modelers who have a fitted SM (from SmoreBase) and real-world data, and want to calibrate CM parameters against those observations.

**Status:** Implemented. See [README.md](README.md) for user-facing usage; this document
records the behavioral spec and acceptance criteria.

---

### Feature: CM Posterior Inference

**One-line description:** Infer a posterior distribution over CM parameter space given real-world data and SM UQ.

**Priority:** Must-have

**Motivation:**
The SM fitting step (SmoreBase) produces SM parameters and their uncertainty for each CM parameter set used to generate training data. The posterior inference step connects this to real-world observations: by sweeping CM parameter space and evaluating how well the SM (within its uncertainty region) fits the data, we can identify which CM parameters are consistent with observations.

**Behavioral specification:**
- `buildPosterior(sm, data, uq_results, cm_sample, cm_prior; ...) -> CMPosteriorResult` —
  primary entry point; also dispatches on `SMFitProblem` for callers already holding one
  - `sm::AbstractSurrogateModel` — the surrogate model used to build `uq_results`
  - `data::AbstractCMData` — real observations as a single-param-set `CMData` (`n_param_sets == 1`)
  - `uq_results::Vector{<:ProfileLikelihoodResult}` — CM-side profiles, one per cohort point (from SmoreBase)
  - `cm_sample::AbstractCMSample` (or a raw `cm_params` matrix, wrapped via `CMSample`) — CM
    parameter locations, **row-aligned** with `uq_results`
  - `cm_prior::ParameterPrior` — CM parameter names and support
- Keyword arguments:
  - `bridge::Symbol = :box_overlap` — overlap test between the two SM-parameter confidence
    regions: `:box_overlap` (symmetric box-vs-box), `:data_trace_in_box` (data profile trace
    points inside the CM box), `:symmetric_trace`
  - `posterior::Symbol = :accept` — `:accept` (hard set) or `:graded` (continuous scores)
  - `profile_options::ProfileLikelihood`, `p0`, `loss`, `acceptance_tol`, `interp`
- Output: `CMPosteriorResult` — stores `accepted` and `scores` per cohort point plus the
  `cm_sample` (grid-aware), the cohort `uq_results`, derived `lb_table`/`ub_table`, and a
  prebuilt bounds interpolator. Accessors: `posteriorSamples`, `posteriorWeights`,
  `acceptedGrid`, `scoreGrid`.
- Interior CM-point queries (no re-fitting required):
  - `posteriorScore(post, θ_cm) -> Float64` and `inPosterior(post, θ_cm; tol = nothing) -> Bool`,
    plus matrix-batch forms. The CM-side SM-parameter CI bounds are interpolated across the CM
    grid (selected by the `interp` kwarg, default `LinearCIInterp()`); the bridge then runs
    against `post.data_profiles`. Restricted to `GridCMSample` + `:box_overlap` /
    `:data_trace_in_box`; `:symmetric_trace` and scattered layouts raise `ArgumentError`.

**Method (the bridge):** The SM is profiled against the real `data`, yielding marginal χ²(1)
confidence information in the same SM-parameter space as the cohort profiles. A cohort point is
consistent with the data when its SM-parameter confidence region overlaps the data's. CM
parameter locations and their profiles are paired by row order (mirroring SmoreGSA's
`runSensitivity`). Adapted from MATLAB SMoRe ParS but fixing: χ²(k)-on-1-D thresholds (we use
marginal χ²(1) CIs), brittle bound extraction (we reuse SmoreBase's interpolated CIs that
degrade to `nothing`), and the six-way `acceptance_method` sprawl (three clean bridge methods).

**Resolved design questions:**
- Real data type: a single-param-set `CMData` — no new type needed.
- Posterior: accepted cohort points (uniform sample set) by default, or normalized scores as a
  graded discrete posterior; grid-aware via the retained `cm_sample`.
- SM uncertainty propagation: each `ProfileCurve`'s marginal CI / in-CI trace defines the
  SM-parameter confidence region used in the overlap test.

**Acceptance criteria:** The cohort point that generated the data is accepted and scores
highest under all three bridge methods; cohort points with disjoint SM regions are rejected;
graded scores decrease away from the best match. Covered in `test/runtests.jl`.

**Future directions (not implemented):**
- Profile smoothing/refinement before bridging.
- Arc-length / Jacobian-weighted trace scoring (the point fraction is reparameterization-
  dependent).
- A trace-distance "street area" consistency measure.
- **Cohort refinement diagnostic.** Linear bound interpolation is reliable only when the bound
  surface moves a small fraction of its own width between adjacent cohort points — a
  dimensionless ratio comparing SM-space shift to SM-space width (the CM step is folded into
  the finite-difference numerator; both pieces of the ratio carry SM-parameter units). Three
  different causes make that ratio large — steep manifold, tight CIs, or coarse CM cohort —
  and the interpolator can't tell them apart; in each case interior queries can return zero
  even near a cohort point that is in the posterior. The user's only knob is cohort density,
  so the corrective action is identical in all three. The diagnostic should report the
  per-edge ratio (and a second-difference / curvature variant for axes with ≥ 3 points) so
  users can add cohort points where flagged.
- Profile-shape-aware interior interpolation (a richer alternative to mere refinement: use the
  full profile LL curve, not just `(lb, ub)`, in the interpolation).
