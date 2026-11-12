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

---

## Session: `buildPosterior` drops `cm_prior` (2026-07-02)

### Goal
Flagged during a cross-repo SmoreBase review session: `cm_prior::ParameterPrior` was a required
`buildPosterior` argument used **only** to extract `.names` — a whole prior (bounds +
distributions) just to label the posterior. SmoreBase's `GridCMSample`/`ScatteredCMSample` were
extended in the same session to carry `names` natively (see SmoreBase's "CM-sample unification"
entry above, and its own progress.md "ODE SM refinements + batched profile-likelihood UQ"
session), so the redundancy could be closed here.

### Decision
Drop `cm_prior` from all 4 `buildPosterior` methods. The `cm_sample`-input methods gain
`cm_names::Vector{String} = cm_sample.names`; the raw-matrix-input methods gain
`cm_names::Union{Nothing,Vector{String}} = nothing`, threaded into
`CMSample(cm_params; names = cm_names)` and left to the delegated call's own default to pick up
from there (no double-specification of the same default logic in two places).

Considered keeping `cm_prior` as an optional override alongside `cm_names` — rejected as two
knobs for one job; `cm_names` alone (defaulting from `cm_sample`) covers every case `cm_prior`
did here, since SmoreFit never used `cm_prior`'s distributions (unlike SmoreGSA's
`runSensitivity`, which genuinely needs them for inverse-CDF sampling and keeps its own
`cm_prior` argument unchanged).

### Status
Implemented on `feature/build-posterior-cm-names`. ~16 call sites in `test/runtests.jl` updated
(dropped trailing `cm_prior`); new `cm_names` sub-testset added. Depends on the SmoreBase
`names` field landing first (local `dev`-linked path dependency, no version bump needed).

---

## Session: `cm_param_set` rename (2026-07-02)

### Goal
SmoreBase renamed `param_set` → `cm_param_set` throughout (see its progress.md) to disambiguate
CM parameter vectors from SM parameters once a single call can involve both. SmoreFit called the
same concept "cohort"/"cohort point" in its own docs and had two internal helpers
(`_cohortMLE`, `_meanCohortMLE`) named after that term — updated to match.

### Decision
- Calls into SmoreBase's renamed API updated: `n_param_sets` → `n_cm_param_sets`, `CMData(...;
  param_sets=...)` → `cm_param_sets=` in tests.
- Internal helpers renamed: `_cohortMLE` → `_cmParamSetMLE`, `_meanCohortMLE` →
  `_meanCmParamSetMLE`.
- "Cohort"/"cohort point" prose replaced with "CM param_set" throughout `posterior.jl`,
  `bridge.jl`, `interior.jl`, `data_profiles.jl`, `types/posterior_result.jl`, PRD.md, README.md.
- **Caught during testing:** the mechanical rename briefly introduced a local variable named
  `n_cm_param_sets` inside `buildPosterior` (from a pre-existing `n_cohort` local) that shadowed
  the imported `SmoreBase.n_cm_param_sets` function in the same scope — Julia's whole-function
  scoping rules turned the earlier `n_cm_param_sets(problem.data)` call into an
  `UndefVarError`. Renamed the local to `n_ps` (matching the naming already used for the same
  quantity in SmoreBase's `fitting.jl`).

### Status
Implemented on `feature/build-posterior-cm-names` (same branch as the `cm_names` work above,
since both are part of the same terminology cleanup pass). Full test suite green (82 tests).

---

## Session: Copilot review fixes on PR #9 (2026-07-02)

### Goal
Copilot flagged 4 issues on the open PR: three "CM cm_param_set"-style duplicate-prefix rename
artifacts (README.md, `types/posterior_result.jl`, `bridge.jl` — the same class of bug as
SmoreBase's own rename sweep, just missed here), and one real gap: `buildPosterior`'s `cm_names`
kwarg was never validated against the number of CM parameters (`size(cm_sample.params, 2)`), so
a mismatched length would silently mislabel the posterior or blow up downstream in a consumer
rather than failing at the source.

### Decision
- Fixed the three duplicate-"CM" prose spots.
- Added a length check for `cm_names` in `buildPosterior`'s primary method, right alongside the
  existing `uq_results`/`cm_sample` length check, throwing `ArgumentError` with the mismatched
  lengths reported.
- Added regression tests: mismatched `cm_names` on both the `cm_sample`-input and raw-matrix-input
  forms.

### Status
Implemented on `feature/build-posterior-cm-names`. Full test suite green (84 tests).

---

## Session: Drop dead `posterior` field/kwarg; rename `acceptance_tol`/`tol` → `min_score` (2026-07-02)

### Goal
Review of `CMPosteriorResult` surfaced two naming/API issues: the `posterior::Symbol`
kwarg/field (`:accept`/`:graded`) is validated and stored but never consulted by any
computation — `posteriorSamples` always reads `accepted`, `posteriorWeights` always reads
`scores`, regardless of its value — so it's dead weight rather than a real mode switch.
Separately, `acceptance_tol` (on `buildPosterior`/`CMPosteriorResult`) and `tol` (on
`inPosterior`) name a *score* threshold where a **lower** value accepts *more* cm_param_sets —
backwards from what "tolerance" implies.

### Decision
- Dropped `posterior` entirely: no kwarg on any `buildPosterior` method, no field on
  `CMPosteriorResult`. Which view to use is now purely a choice of accessor
  (`posteriorSamples` vs. `posteriorWeights`), which was already true in practice.
- Renamed `acceptance_tol` → `min_score` throughout (kwarg, field, docstrings, README, PRD).
- Renamed `inPosterior`'s `tol` kwarg → `min_score` to match, defaulting directly to
  `post.min_score` rather than `nothing` + a ternary (simpler now that both positional- and
  keyword-argument forms can reference `post` in the default expression).
- Removed the now-meaningless "unknown posterior mode" validation test.

### Status
Implemented on `feature/drop-posterior-field-rename-min-score`. Full test suite green (82
tests, one fewer than before since the dead `posterior`-validation test was removed).

---

## Session: Jaccard box score — opt-in in v0.5.1, default in v0.6.0 (2026-11-12)

### Goal
`_boxOverlapScore` divides the overlap volume by the **data box's** volume. The docs called it
symmetric; it is not, and a CM box that contains the data box scores 1.0 however large it is.
Open profile CIs make that common: `_profileBox` stretches an open side to the swept-range edge,
so a non-identifiable CM param_set gets the maximum score. In the bridge-paper Tier 1 study
(SmorePaper, `analysis/tier1_round/notes.md`) all 27 open-CI false accepts scored 1.0; as
Jaccard they score ≤ 0.006, and the bridge's rank agreement with both ground truths goes from
negative to 0.74–0.93.

### Decision
- Score `:box_overlap` as the **Jaccard index** (overlap ÷ union). A choice of score for the
  existing bridge, not a new bridge: new `box_score` keyword (`:jaccard` / `:data_normalized`).
- Two releases (Daniel's call). **v0.5.1:** opt-in; unset → `:data_normalized` plus a one-time
  `@warn` (`maxlog = 1`, visible by default — `Base.depwarn` is hidden by default). **v0.6.0:**
  default `:jaccard`, warning removed, `:data_normalized` stays available.
- `box_score` is stored on `CMPosteriorResult` and threaded through `_consistency` /
  `_consistencyBox` as an argument, so interior queries reproduce stored scores. The 11-argument
  positional constructor is kept (defaults `:data_normalized`) so direct construction does not
  break in a patch release.
- Zero-width dimensions: the handoff proposed keeping `_boxOverlapScore`'s containment test
  (drop a contained zero-width data dimension). Rejected in review: it gave a data point inside a
  wide CM box a score of 1.0 (and the swapped pair 0), which breaks symmetry and rewards width.
  Jaccard treats a box with a zero-width dimension as having zero volume, so the score is 0
  (also when both are zero-width at the same point, 0 ÷ 0 → 0). A zero-width CI should only come
  from a degenerate profile (e.g. −Inf log-likelihood beside the MLE, or a zero-width sweep), so
  `buildPosterior` now warns whenever one appears.
- `_boxOverlapScore` is unchanged byte-for-byte; `_boxJaccardScore` sits beside it.
- v0.6.0 drops the `nothing` sentinel and the warning, and also the 11-argument positional
  constructor. That constructor was kept in v0.5.1 only so a patch release would not break
  direct construction. In v0.6 only `buildPosterior` builds a `CMPosteriorResult`, so direct
  construction is not public API, and v0.6.0 is breaking anyway.

### Testing note
In the existing test fixture every CM box lies *inside* the data box (the data profile is wider
than the CM-side ones), where union = data box and the two scores coincide. The end-to-end test
therefore adds a CM param_set with 5× CM noise, whose box contains the data box: data-normalized
1.0, Jaccard ≈ 0.03.

### Out of scope (separate candidate changes)
The open-CI rule itself, the trace bridges' scores, profile-point spacing, the data-profile
defaults, prior-weighted or log-coordinate volume.
