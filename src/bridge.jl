# Bridging two sets of SM-parameter profiles.
#
# Both the CM (per cm_param_set) and the real data, through the shared surrogate model,
# induce confidence regions in the same SM-parameter space. A CM parameter set is consistent
# with the data iff its SM region overlaps the data's SM region. Profiles only give a
# 1-D-per-parameter view, so we approximate each region either as a hyperrectangle ("box")
# from the marginal χ²(1) CIs, or as the union of profile trace points within those CIs.

# Hyperrectangle (per-parameter marginal CI) for a profile likelihood result.
# A `nothing` CI bound (parameter unidentified on that side) falls back to the swept-range
# extreme — the SM prior/sweep edge — rather than ±Inf, so the prior still bounds the region.
function _profileBox(plr::ProfileLikelihoodResult)
    k  = length(plr.profiles)
    lo = Vector{Float64}(undef, k)
    hi = Vector{Float64}(undef, k)
    for (j, pc) in enumerate(plr.profiles)
        lo[j] = pc.ci_lower === nothing ? first(pc.profile_values) : pc.ci_lower
        hi[j] = pc.ci_upper === nothing ? last(pc.profile_values)  : pc.ci_upper
    end
    return lo, hi
end

# Union of SM-parameter vectors traced by every profile, restricted to the in-CI portion of
# each path (log-likelihood at or above the profile threshold). Returns `[n_points × k]`.
# Aggregation across parameters = pooling the points of all k profile traces.
function _profileTrace(plr::ProfileLikelihoodResult)
    k    = length(plr.profiles)
    rows = Vector{Vector{Float64}}()
    for pc in plr.profiles
        for r in findall(pc.log_likelihoods .>= pc.threshold)
            push!(rows, pc.optimal_parameters[r, :])
        end
    end
    out = Matrix{Float64}(undef, length(rows), k)
    for (i, row) in enumerate(rows)
        out[i, :] = row
    end
    return out
end

_pointInBox(p, lo, hi) = all(lo .<= p .<= hi)

# Fraction of trace points (rows) lying inside the box [lo, hi].
function _fractionInBox(trace::AbstractMatrix, lo, hi)
    n = size(trace, 1)
    n == 0 && return 0.0
    cnt = count(i -> _pointInBox(view(trace, i, :), lo, hi), 1:n)
    return cnt / n
end

# Relative overlap volume of two boxes, normalized by the data box's width per dimension.
# A degenerate data dimension (`width == 0`) is treated as a point: it contributes factor 1
# when the CM box contains it and short-circuits to 0 otherwise. Non-degenerate disjoint
# dimensions also short-circuit to 0.
function _boxOverlapScore(cm_lo, cm_hi, d_lo, d_hi)
    score = 1.0
    for j in eachindex(cm_lo)
        width = d_hi[j] - d_lo[j]
        if width == 0
            # Degenerate data dimension: factor 1 iff the data point is inside the CM box.
            # We must handle this *before* the overlap test, since overlap == 0 there would
            # otherwise mis-flag a containing CM box as disjoint.
            (cm_lo[j] <= d_lo[j] <= cm_hi[j]) || return 0.0
            continue
        end
        overlap = min(cm_hi[j], d_hi[j]) - max(cm_lo[j], d_lo[j])
        overlap <= 0 && return 0.0
        score *= overlap / width
    end
    return score
end

# Consistency score in [0, 1] given a CM-side box only (no CM-side trace). Box-based bridges
# work directly; trace-based ones cannot — used by interior queries where the CM-side box is
# interpolated across CM space but there is no per-point CM trace to interpolate.
function _consistencyBox(
    cm_lo::AbstractVector,
    cm_hi::AbstractVector,
    data_plr::ProfileLikelihoodResult,
    bridge::Symbol,
)
    if bridge === :box_overlap
        d_lo, d_hi = _profileBox(data_plr)
        return _boxOverlapScore(cm_lo, cm_hi, d_lo, d_hi)
    elseif bridge === :data_trace_in_box
        return _fractionInBox(_profileTrace(data_plr), cm_lo, cm_hi)
    elseif bridge === :symmetric_trace
        throw(ArgumentError(
            ":symmetric_trace requires a CM-side trace and is not supported here"
        ))
    else
        throw(ArgumentError(
            "unknown bridge method :$bridge; expected :box_overlap, :data_trace_in_box, or :symmetric_trace"
        ))
    end
end

# Consistency score in [0, 1] between a CM param_set's profiles and the data's profiles.
function _consistency(cm_plr::ProfileLikelihoodResult, data_plr::ProfileLikelihoodResult, bridge::Symbol)
    cm_lo, cm_hi = _profileBox(cm_plr)
    if bridge === :symmetric_trace
        d_lo, d_hi = _profileBox(data_plr)
        return max(
            _fractionInBox(_profileTrace(data_plr), cm_lo, cm_hi),
            _fractionInBox(_profileTrace(cm_plr),   d_lo, d_hi),
        )
    else
        return _consistencyBox(cm_lo, cm_hi, data_plr, bridge)
    end
end
