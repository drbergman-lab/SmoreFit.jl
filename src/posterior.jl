"""
    buildPosterior(problem::SMFitProblem, uq_results, cm_sample, cm_prior; kwargs...) -> CMPosteriorResult
    buildPosterior(problem::SMFitProblem, uq_results, cm_params::AbstractMatrix, cm_prior; kwargs...)
    buildPosterior(sm, data, uq_results, cm_sample, cm_prior; loss=GaussianNLL(), kwargs...)
    buildPosterior(sm, data, uq_results, cm_params::AbstractMatrix, cm_prior; loss=GaussianNLL(), kwargs...)

Infer which CM parameter sets are consistent with real-world data, given the surrogate model
and its per-cohort profile likelihood UQ from SmoreBase.

Two streams of profile information are combined:

1. **CM-side, already done** — `uq_results[i]` is the SM's profile likelihood against the CM
   cohort point at row `i` of `cm_sample.params`. These were produced upstream (e.g. by
   SmoreBase's `quantifyUncertainty` over the cohort) and are passed in unchanged.
2. **Data-side, computed here** — the SM is fit and then profiled against the real
   observations. The ingredients for *that* fit-and-profile step are what `problem` (or the
   `(sm, data, loss)` triple, with `sm_prior` taken from `uq_results[1].fit_result.prior`)
   describe — **not** anything about the CM cohort, which is fully captured by `uq_results`.

Each cohort point is then scored by how well its SM-parameter confidence region (from
`uq_results[i]`) overlaps the data's SM-parameter confidence region (from the data profile),
via the chosen `bridge`. Consistent cohort points form the posterior over CM parameter space.

The CM-side profiles and CM parameters are **row-aligned**: `uq_results[i]` ↔
`cm_sample.params[i, :]` (mirroring SmoreGSA's `runSensitivity`). A raw matrix is accepted and
wrapped with `CMSample` (from SmoreBase: grid if possible, else scattered).

# Arguments — describing how to fit/profile the SM against the *real data*
- `problem::SMFitProblem` — bundles `sm`, the real `data` (single param-set), the SM prior,
  and the loss to use when fitting and profiling the SM against that data
- *or* `sm::AbstractSurrogateModel` + `data::AbstractCMData` — equivalent to passing a problem
  built internally as `SMFitProblem(sm, data, uq_results[1].fit_result.prior; loss = loss)`

# Arguments — describing the CM cohort already profiled upstream
- `uq_results::Vector{<:ProfileLikelihoodResult}` — one profile likelihood result per cohort
  point, computed upstream against the cohort's CM-generated data
- `cm_sample::AbstractCMSample` (or `cm_params::AbstractMatrix`) — CM parameter locations,
  row-aligned with `uq_results`
- `cm_prior::ParameterPrior` — CM parameter names and support (for labeling the posterior)

# Keyword Arguments
- `bridge::Symbol = :box_overlap` — overlap test: `:box_overlap` (symmetric box-vs-box),
  `:data_trace_in_box` (data profile trace points inside the CM box), or `:symmetric_trace`
- `posterior::Symbol = :accept` — `:accept` (hard accept/reject set) or `:graded` (use scores)
- `profile_options::ProfileLikelihood = ProfileLikelihood()` — settings for the **data**
  profile (the CM-side profiles in `uq_results` are already final and not recomputed)
- `p0 = nothing` — initial guess for the **data** fit (`[1 × n_sm_params]`); defaults to the
  column-mean of the cohort SM fits
- `loss::AbstractLoss = GaussianNLL()` — loss for the **data** fit, *only on the
  `(sm, data, …)` forms*; the problem form uses `problem.loss`
- `acceptance_tol::Real = 0.0` — a cohort point is accepted iff its score exceeds this; also
  the default threshold for [`inPosterior`](@ref) interior queries
- `interp::AbstractCIInterpolator = LinearCIInterp()` — how the per-cohort SM-parameter CI
  bounds are interpolated across the CM grid for interior queries; only consulted when
  `cm_sample isa GridCMSample` (interior queries are not built for scattered layouts)

# Returns
[`CMPosteriorResult`](@ref). Use [`posteriorSamples`](@ref) for accepted CM sets,
[`posteriorWeights`](@ref) for a graded posterior, and [`acceptedGrid`](@ref) /
[`scoreGrid`](@ref) for grid-shaped views.

# Example
```julia
# If you already have an SMFitProblem for the real data in hand:
problem = SMFitProblem(sm, data, sm_prior)              # SM + real data + SM prior
post    = buildPosterior(problem, uq_results, cm_params, cm_prior; bridge = :box_overlap)

# Otherwise, pass sm and the real data directly:
post = buildPosterior(sm, data, uq_results, cm_params, cm_prior)

posteriorSamples(post)   # CM parameter vectors consistent with the data
```
"""
function buildPosterior(
    problem::SMFitProblem,
    uq_results::Vector{<:ProfileLikelihoodResult},
    cm_sample::AbstractCMSample,
    cm_prior::ParameterPrior;
    bridge::Symbol = :box_overlap,
    posterior::Symbol = :accept,
    profile_options::ProfileLikelihood = ProfileLikelihood(),
    p0 = nothing,
    acceptance_tol::Real = 0.0,
    interp::AbstractCIInterpolator = LinearCIInterp(),
)
    n_param_sets(problem.data) == 1 || throw(ArgumentError(
        "`problem.data` must have a single param set (n_param_sets == 1), got $(n_param_sets(problem.data))"
    ))
    n_cohort = size(cm_sample.params, 1)
    length(uq_results) == n_cohort || throw(ArgumentError(
        "length(uq_results)=$(length(uq_results)) must equal the number of cohort points $n_cohort"
    ))
    bridge in (:box_overlap, :data_trace_in_box, :symmetric_trace) || throw(ArgumentError(
        "unknown bridge :$bridge; expected :box_overlap, :data_trace_in_box, or :symmetric_trace"
    ))
    posterior in (:accept, :graded) || throw(ArgumentError(
        "unknown posterior :$posterior; expected :accept or :graded"
    ))

    p0_eff   = p0 === nothing ? _meanCohortMLE(uq_results) : p0
    data_plr = _profileAgainstData(problem, p0_eff, profile_options)

    scores   = Float64[_consistency(uq, data_plr, bridge) for uq in uq_results]
    accepted = BitVector(scores .> acceptance_tol)

    # Per-cohort SM-parameter CI tables, used both to build the grid bounds interpolant for
    # interior queries and stored for later inspection / re-analysis. Uses the same swept-edge
    # fallback for `nothing` CI bounds as the `:box_overlap` bridge does internally.
    n_sm      = length(uq_results[1].profiles)
    lb_table  = Matrix{Float64}(undef, n_cohort, n_sm)
    ub_table  = Matrix{Float64}(undef, n_cohort, n_sm)
    for (i, uq) in enumerate(uq_results)
        lo, hi = _profileBox(uq)
        lb_table[i, :] = lo
        ub_table[i, :] = hi
    end

    # Interior queries only make sense on a grid layout (and the v1 supported bridges); for
    # other layouts we skip building the interpolator, and the interior API errors clearly.
    get_bounds = if cm_sample isa GridCMSample
        SmoreBase._buildBoundsInterpolant(cm_sample, lb_table, ub_table, interp)
    else
        nothing
    end

    return CMPosteriorResult(
        cm_sample, cm_prior.names, accepted, scores, bridge, posterior,
        Float64(acceptance_tol), data_plr,
        Vector{ProfileLikelihoodResult}(uq_results), lb_table, ub_table, get_bounds,
    )
end

# problem + raw cm_params matrix
function buildPosterior(
    problem::SMFitProblem,
    uq_results::Vector{<:ProfileLikelihoodResult},
    cm_params::AbstractMatrix,
    cm_prior::ParameterPrior;
    kwargs...,
)
    return buildPosterior(problem, uq_results, CMSample(cm_params), cm_prior; kwargs...)
end

# sm + data convenience: build the SMFitProblem (using uq_results' SM prior) and delegate.
function buildPosterior(
    sm::AbstractSurrogateModel,
    data::AbstractCMData,
    uq_results::Vector{<:ProfileLikelihoodResult},
    cm_sample::AbstractCMSample,
    cm_prior::ParameterPrior;
    loss::AbstractLoss = GaussianNLL(),
    kwargs...,
)
    sm_prior = uq_results[1].fit_result.prior
    problem  = SMFitProblem(sm, data, sm_prior; loss = loss)
    return buildPosterior(problem, uq_results, cm_sample, cm_prior; kwargs...)
end

# sm + data + raw cm_params matrix
function buildPosterior(
    sm::AbstractSurrogateModel,
    data::AbstractCMData,
    uq_results::Vector{<:ProfileLikelihoodResult},
    cm_params::AbstractMatrix,
    cm_prior::ParameterPrior;
    kwargs...,
)
    return buildPosterior(sm, data, uq_results, CMSample(cm_params), cm_prior; kwargs...)
end
