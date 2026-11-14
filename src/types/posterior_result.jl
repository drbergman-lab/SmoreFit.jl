"""
    CMPosteriorResult

Result of [`buildPosterior`](@ref): which CM param_sets are consistent with the
real-world data, scored by the chosen bridge method.

Both a hard accept/reject set (`accepted`) and the continuous `scores` are always stored; which
one to use is a choice made at read time via [`posteriorSamples`](@ref) (accept/reject) or
[`posteriorWeights`](@ref) (graded). Because `cm_sample` is retained (with its grid `axes` for a
`GridCMSample`), acceptance and scores can be reshaped back onto the CM grid via
[`acceptedGrid`](@ref) / [`scoreGrid`](@ref).

For grid-aware results, the per-cm_param_set SM-parameter CI tables (`lb_table`, `ub_table`) and a
prebuilt CM-grid bounds interpolator (`get_bounds`) are stored so that interior CM points (not
among the cm_param_sets) can be queried via [`inPosterior`](@ref) and [`posteriorScore`](@ref). The full
`uq_results` are also retained for inspection and any future re-analysis.

The result also records what built it: the data-side problem (`data_problem`), both sides'
profile settings (`data_profiles.method` and each `uq_results[i].method`), the bridge
settings (`bridge`, `box_score`, `min_score`) and the interpolation method (`interp`).

# Fields
- `cm_sample` — CM parameter points (`AbstractCMSample`), row-aligned with `accepted`/`scores`
- `cm_names` — CM parameter names (defaults from `cm_sample.names`; override via `buildPosterior`'s `cm_names` kwarg)
- `accepted` — `BitVector`; `true` where the cm_param_set's consistency score exceeds `min_score`
- `scores` — consistency score per cm_param_set, in `[0, 1]`
- `bridge` — the bridge method used (`:box_overlap`, `:data_trace_in_box`, `:symmetric_trace`)
- `min_score` — threshold used for `accepted`; default for interior queries
- `data_problem` — the `SMFitProblem` (`sm`, real `data`, SM `prior`, `loss`) that the SM was
  fitted and profiled against on the data side; for the `(sm, data, …)` methods, the problem
  they build internally
- `data_profiles` — the SM profile likelihood computed against the real data. Its `method`
  field holds the data-side profile settings (`buildPosterior`'s `profile_options`), and
  `data_profiles.fit_result.initial_parameters` is the data fit's starting point (`p0`)
- `uq_results` — the CM param_sets' profiles, one per row of `cm_sample.params`. Each
  `uq_results[i].method` holds the CM-side profile settings it was computed with (`nothing`
  for a result assembled by hand)
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
  reproduce the stored `scores`. Recorded for every bridge but only consulted by `:box_overlap`.

The data-side SM-parameter box and trace that the bridges compared against are not stored;
they are `SmoreFit._profileBox(post.data_profiles)` and `SmoreFit._profileTrace(post.data_profiles)`.

# See also
[`posteriorSamples`](@ref), [`posteriorWeights`](@ref), [`acceptedGrid`](@ref),
[`scoreGrid`](@ref), [`inPosterior`](@ref), [`posteriorScore`](@ref).
"""
struct CMPosteriorResult
    cm_sample      :: AbstractCMSample
    cm_names       :: Vector{String}
    accepted       :: BitVector
    scores         :: Vector{Float64}
    bridge         :: Symbol
    min_score      :: Float64
    data_problem   :: SMFitProblem
    data_profiles  :: ProfileLikelihoodResult
    uq_results     :: Vector{ProfileLikelihoodResult}
    lb_table       :: Matrix{Float64}
    ub_table       :: Matrix{Float64}
    interp         :: AbstractCIInterpolator
    get_bounds     :: Union{Nothing,Function}
    box_score      :: Symbol
end

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
distribution) over the CM param_sets. All-zero if no CM param_set is consistent.

# Example
```julia
post = buildPosterior(sm, data, uq_results, cm_params)
w    = posteriorWeights(post)   # weight per cm_param_set, sums to 1
```
"""
function posteriorWeights(r::CMPosteriorResult)
    s = sum(r.scores)
    s == 0 && return zeros(Float64, length(r.scores))
    return r.scores ./ s
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

Reshape `scores` onto the CM parameter grid. Requires a `GridCMSample` layout.

# Example
```julia
post = buildPosterior(sm, data, uq_results, cm_params)   # cm_params a regular grid
scoreGrid(post)   # Float64 array of size length.(cm_sample.axes)
```
"""
scoreGrid(r::CMPosteriorResult) = reshapeToGrid(r.cm_sample, r.scores)
