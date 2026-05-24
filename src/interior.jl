# Interior CM-point queries on CMPosteriorResult.
#
# The cohort defines a posterior at the sampled CM points. For an interior CM point (not in
# the cohort), we interpolate the per-cohort SM-parameter CI bounds across the CM grid via
# SmoreBase's `_buildBoundsInterpolant`, then evaluate the chosen bridge against the data
# profile at the interpolated box. v1 supports `GridCMSample` + `:box_overlap` /
# `:data_trace_in_box` only; `:symmetric_trace` and non-grid layouts error clearly.

function _requireInteriorSupport(post::CMPosteriorResult)
    post.cm_sample isa GridCMSample || throw(ArgumentError(
        "interior queries require a GridCMSample layout, got $(typeof(post.cm_sample))"
    ))
    post.bridge in (:box_overlap, :data_trace_in_box) || throw(ArgumentError(
        "interior queries are not supported for bridge :$(post.bridge); " *
        "use :box_overlap or :data_trace_in_box"
    ))
    post.get_bounds === nothing && throw(ArgumentError(
        "this CMPosteriorResult has no bounds interpolator (was it built with a non-grid layout?)"
    ))
    return nothing
end

"""
    posteriorScore(post::CMPosteriorResult, θ_cm::AbstractVector) -> Float64
    posteriorScore(post::CMPosteriorResult, queries::AbstractMatrix) -> Vector{Float64}

Consistency score at one or more interior CM parameter points, using the same bridge that
produced `post`. The CM-side SM-parameter CI bounds are interpolated across the CM grid; the
data-side profile (`post.data_profiles`) is unchanged.

`queries[k, :]` is the k-th query point; the vector form is one query.

Requires `post.cm_sample isa GridCMSample` and `post.bridge in (:box_overlap, :data_trace_in_box)`.

# Example
```julia
post = buildPosterior(sm, data, uq_results, cm_params, cm_prior)   # cm_params a regular grid
posteriorScore(post, [θ1, θ2])
posteriorScore(post, [θ1_a θ2_a; θ1_b θ2_b])   # batch
```
"""
function posteriorScore(post::CMPosteriorResult, θ_cm::AbstractVector)
    _requireInteriorSupport(post)
    lb, ub = post.get_bounds(θ_cm)
    return _consistencyBox(lb, ub, post.data_profiles, post.bridge)
end

function posteriorScore(post::CMPosteriorResult, queries::AbstractMatrix)
    _requireInteriorSupport(post)
    n = size(queries, 1)
    out = Vector{Float64}(undef, n)
    for k in 1:n
        lb, ub = post.get_bounds(@view queries[k, :])
        out[k] = _consistencyBox(lb, ub, post.data_profiles, post.bridge)
    end
    return out
end

"""
    inPosterior(post::CMPosteriorResult, θ_cm::AbstractVector; tol = nothing) -> Bool
    inPosterior(post::CMPosteriorResult, queries::AbstractMatrix; tol = nothing) -> BitVector

Is the interior CM parameter point in the posterior? Equivalent to `posteriorScore(post, …) >
tol`. `tol === nothing` (default) uses `post.acceptance_tol`.

Requires `post.cm_sample isa GridCMSample` and `post.bridge in (:box_overlap, :data_trace_in_box)`.

# Example
```julia
post = buildPosterior(sm, data, uq_results, cm_params, cm_prior)   # cm_params a regular grid
inPosterior(post, [θ1, θ2])
inPosterior(post, queries; tol = 0.05)
```
"""
function inPosterior(
    post::CMPosteriorResult,
    θ_cm::AbstractVector;
    tol::Union{Nothing,Real} = nothing,
)
    threshold = tol === nothing ? post.acceptance_tol : tol
    return posteriorScore(post, θ_cm) > threshold
end

function inPosterior(
    post::CMPosteriorResult,
    queries::AbstractMatrix;
    tol::Union{Nothing,Real} = nothing,
)
    threshold = tol === nothing ? post.acceptance_tol : tol
    return BitVector(posteriorScore(post, queries) .> threshold)
end
