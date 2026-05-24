"""
    CMPosteriorResult

Result of [`buildPosterior`](@ref): which CM cohort parameter sets are consistent with the
real-world data, scored by the chosen bridge method.

Both a hard accept/reject set (`accepted`) and the continuous `scores` are always stored; the
`posterior` field records which one the caller asked to emphasize. Because `cm_sample` is
retained (with its grid `axes` for a `GridCMSample`), acceptance and scores can be
reshaped back onto the CM grid via [`acceptedGrid`](@ref) / [`scoreGrid`](@ref).

For grid-aware results, the per-cohort SM-parameter CI tables (`lb_table`, `ub_table`) and a
prebuilt CM-grid bounds interpolator (`get_bounds`) are stored so that interior CM points (not
in the cohort) can be queried via [`inPosterior`](@ref) and [`posteriorScore`](@ref). The full
`uq_results` are also retained for inspection and any future re-analysis.

# Fields
- `cm_sample` — CM parameter points (`AbstractCMSample`), row-aligned with `accepted`/`scores`
- `cm_names` — CM parameter names (from `cm_prior`)
- `accepted` — `BitVector`; `true` where the cohort point's consistency score exceeds `acceptance_tol`
- `scores` — consistency score per cohort point, in `[0, 1]`
- `bridge` — the bridge method used (`:box_overlap`, `:data_trace_in_box`, `:symmetric_trace`)
- `posterior` — `:accept` or `:graded`
- `acceptance_tol` — threshold used for `accepted`; default for interior queries
- `data_profiles` — the SM profile likelihood computed against the real data (kept for inspection)
- `uq_results` — the cohort CM-side profiles, one per row of `cm_sample.params`
- `lb_table`, `ub_table` — `[n_cohort × n_sm_params]` per-cohort SM-parameter CI lower/upper bounds
  derived from `uq_results` (with `nothing` falling back to the profile's swept-range extreme,
  matching the `:box_overlap` bridge)
- `get_bounds` — closure `θ_cm -> (lb, ub)` interpolating the CI tables across the CM grid;
  `nothing` for non-grid layouts

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
    posterior      :: Symbol
    acceptance_tol :: Float64
    data_profiles  :: ProfileLikelihoodResult
    uq_results     :: Vector{ProfileLikelihoodResult}
    lb_table       :: Matrix{Float64}
    ub_table       :: Matrix{Float64}
    get_bounds     :: Union{Nothing,Function}
end

"""
    posteriorSamples(r::CMPosteriorResult) -> Matrix

The accepted CM parameter sets, `[n_accepted × n_cm_params]` — the posterior as a uniform
sample set over CM parameter space.

# Example
```julia
post    = buildPosterior(sm, data, uq_results, cm_params, cm_prior)
samples = posteriorSamples(post)   # rows are CM parameter vectors consistent with the data
```
"""
posteriorSamples(r::CMPosteriorResult) = r.cm_sample.params[r.accepted, :]

"""
    posteriorWeights(r::CMPosteriorResult) -> Vector{Float64}

Consistency scores normalized to sum to one — a simple graded posterior (discrete
distribution) over the CM cohort points. All-zero if no cohort point is consistent.

# Example
```julia
post = buildPosterior(sm, data, uq_results, cm_params, cm_prior; posterior = :graded)
w    = posteriorWeights(post)   # weight per cohort point, sums to 1
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
post = buildPosterior(sm, data, uq_results, cm_params, cm_prior)   # cm_params a regular grid
acceptedGrid(post)   # Bool array of size length.(cm_sample.axes)
```
"""
acceptedGrid(r::CMPosteriorResult) = reshapeToGrid(r.cm_sample, r.accepted)

"""
    scoreGrid(r::CMPosteriorResult) -> Array{Float64}

Reshape `scores` onto the CM parameter grid. Requires a `GridCMSample` layout.

# Example
```julia
post = buildPosterior(sm, data, uq_results, cm_params, cm_prior)   # cm_params a regular grid
scoreGrid(post)   # Float64 array of size length.(cm_sample.axes)
```
"""
scoreGrid(r::CMPosteriorResult) = reshapeToGrid(r.cm_sample, r.scores)
