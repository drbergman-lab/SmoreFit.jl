"""
    CMPosteriorResult

Result of [`buildPosterior`](@ref): which CM param_sets are consistent with the
real-world data, scored by the chosen bridge method.

Both a hard accept/reject set (`accepted`) and the continuous `raw_scores` are always stored,
together with which CM param_sets pass the gate (`gate_pass`); which view to use is a choice made at read time via [`posteriorSamples`](@ref) (accept/reject) or
[`posteriorWeights`](@ref) (graded). Because `cm_sample` is retained (with its grid `axes` for a
`GridCMSample`), acceptance and scores can be reshaped back onto the CM grid via
[`acceptedGrid`](@ref) / [`scoreGrid`](@ref).

For grid-aware results, the per-cm_param_set SM-parameter CI tables (`lb_table`, `ub_table`) and a
prebuilt CM-grid bounds interpolator (`get_bounds`) are stored so that interior CM points (not
among the cm_param_sets) can be queried via [`inPosterior`](@ref) and [`posteriorScore`](@ref). The full
`uq_results` are also retained for inspection and any future re-analysis.

The result also records what built it: the data-side problem (`data_problem`), both sides'
profile settings (`data_profiles.method` and each `uq_results[i].method`), the bridge
settings (`bridge`, `box_score`, `gate`, `min_score`) and the interpolation method (`interp`).

# Fields
- `cm_sample` — CM parameter points (`AbstractCMSample`), row-aligned with `accepted`/`raw_scores`/`gate_pass`
- `cm_names` — CM parameter names (defaults from `cm_sample.names`; override via `buildPosterior`'s `cm_names` kwarg)
- `accepted` — `BitVector`; `true` where the cm_param_set passes the gate and its score exceeds
  `min_score`: `gate_pass .& (raw_scores .> min_score)`
- `raw_scores` — the bridge's consistency score per cm_param_set, in `[0, 1]`, before any gate.
  A cm_param_set that fails the gate keeps its raw score here but counts as score 0 everywhere a
  score is used ([`scoreGrid`](@ref), [`posteriorWeights`](@ref), `accepted`, the elbow). The
  scores of the passing cm_param_sets are `raw_scores[gate_pass]`
- `bridge` — the bridge method used (`:box_overlap`, `:data_trace_in_box`, `:symmetric_trace`)
- `min_score` — threshold used for `accepted`; default for interior queries. When
  `buildPosterior` was called with `min_score = :elbow`, this is the resulting τ_elbow
- `data_problem` — the `SMFitProblem` (`sm`, real `data`, SM `prior`, `loss`) that the SM was
  fitted and profiled against on the data side; for the `(sm, data, …)` methods, the problem
  they build internally
- `data_profiles` — the SM profile likelihood computed against the real data. Its `method`
  field holds the data-side profile settings (`buildPosterior`'s `profile_options`), and
  `data_profiles.fit_result.initial_parameters` is the data fit's starting point (`p0`)
- `uq_results` — the CM param_sets' profiles, one per row of `cm_sample.params`. Each
  `uq_results[i].method` holds the CM-side profile settings it was computed with
- `lb_table`, `ub_table` — `[n_cm_param_sets × n_sm_params]` per-cm_param_set SM-parameter CI lower/upper bounds
  derived from `uq_results` (with `nothing` falling back to the profile's swept-range extreme,
  matching the `:box_overlap` bridge)
- `interp` — the `AbstractCIInterpolator` used to build `get_bounds` (stored for every layout)
- `get_bounds` — callable `θ_cm -> (lb, ub)` interpolating the CI tables across the CM grid
  (a `SmoreBase.CIBoundsInterpolant`); `nothing` for non-grid layouts. It can be rebuilt from
  the stored fields: `SmoreBase._buildBoundsInterpolant(post.cm_sample, post.lb_table,
  post.ub_table, post.interp)`
- `box_score` — how the `:box_overlap` bridge scores box overlap, as passed to
  `buildPosterior`: `:jaccard` (overlap volume ÷ union volume; the default) or
  `:data_normalized` (overlap volume ÷ data-box volume). Interior queries reuse it, so they
  reproduce the stored `raw_scores`. Recorded for every bridge but only consulted by `:box_overlap`.
- `gate` — the [`AbstractGate`](@ref) of this posterior ([`NullGate`](@ref) when none was
  asked for). It is chosen once, in `buildPosterior`; interior queries
  ([`posteriorScore`](@ref), [`inPosterior`](@ref)) and [`applyGate`](@ref) use it
- `gate_pass` — `BitVector`; `true` where the cm_param_set passes `gate` (all `true` under
  [`NullGate`](@ref))

The interpolator and the gate are type parameters: `CMPosteriorResult{I,G}` with
`I <: AbstractCIInterpolator` and `G <: AbstractGate`.

The data-side SM-parameter box and trace that the bridges compared against are not stored;
they are `SmoreFit._profileBox(post.data_profiles)` and `SmoreFit._profileTrace(post.data_profiles)`.

# See also
[`posteriorSamples`](@ref), [`posteriorWeights`](@ref), [`acceptedGrid`](@ref),
[`scoreGrid`](@ref), [`inPosterior`](@ref), [`posteriorScore`](@ref).
"""
struct CMPosteriorResult{I<:AbstractCIInterpolator,G<:AbstractGate}
    cm_sample      :: AbstractCMSample
    cm_names       :: Vector{String}
    accepted       :: BitVector
    raw_scores     :: Vector{Float64}
    bridge         :: Symbol
    min_score      :: Float64
    data_problem   :: SMFitProblem
    data_profiles  :: ProfileLikelihoodResult
    uq_results     :: Vector{ProfileLikelihoodResult}
    lb_table       :: Matrix{Float64}
    ub_table       :: Matrix{Float64}
    interp         :: I
    get_bounds     :: Union{Nothing,Function}
    box_score      :: Symbol
    gate           :: G
    gate_pass      :: BitVector
end

# Scores aligned with the CM param_sets, with 0 where a CM param_set fails the gate. Internal:
# everything that needs one score per CM param_set goes through this.
_gatedScores(r::CMPosteriorResult) = ifelse.(r.gate_pass, r.raw_scores, 0.0)

"""
    posteriorSamples(r::CMPosteriorResult) -> Matrix

The accepted CM parameter sets, `[n_accepted × n_cm_params]` — the posterior as a uniform
sample set over CM parameter space.

# Example
```julia
post    = buildPosterior(sm, data, uq_results, cm_params)
samples = posteriorSamples(post)   # rows are CM parameter vectors consistent with the data
```
"""
posteriorSamples(r::CMPosteriorResult) = r.cm_sample.params[r.accepted, :]

"""
    posteriorWeights(r::CMPosteriorResult) -> Vector{Float64}

Consistency scores normalized to sum to one — a simple graded posterior (discrete
distribution) over the CM param_sets. A CM param_set that fails the gate has weight 0. All-zero
if no CM param_set is consistent.

# Example
```julia
post = buildPosterior(sm, data, uq_results, cm_params)
w    = posteriorWeights(post)   # weight per cm_param_set, sums to 1
```
"""
function posteriorWeights(r::CMPosteriorResult)
    scores = _gatedScores(r)
    s = sum(scores)
    s == 0 && return zeros(Float64, length(scores))
    return scores ./ s
end

"""
    acceptedGrid(r::CMPosteriorResult) -> Array{Bool}

Reshape `accepted` onto the CM parameter grid. Requires a `GridCMSample` layout.

# Example
```julia
post = buildPosterior(sm, data, uq_results, cm_params)   # cm_params a regular grid
acceptedGrid(post)   # Bool array of size length.(cm_sample.axes)
```
"""
acceptedGrid(r::CMPosteriorResult) = reshapeToGrid(r.cm_sample, r.accepted)

"""
    scoreGrid(r::CMPosteriorResult) -> Array{Float64}

Reshape the scores onto the CM parameter grid, with 0 where a CM param_set fails the gate.
Requires a `GridCMSample` layout.

# Example
```julia
post = buildPosterior(sm, data, uq_results, cm_params)   # cm_params a regular grid
scoreGrid(post)   # Float64 array of size length.(cm_sample.axes)
```
"""
scoreGrid(r::CMPosteriorResult) = reshapeToGrid(r.cm_sample, _gatedScores(r))
