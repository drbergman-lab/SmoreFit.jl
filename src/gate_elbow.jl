# The two filters applied to bridge scores before a posterior is read off them:
#   applyGate        scores + a gate → gated scores (a CM param_set that fails the gate scores 0)
#   elbowThreshold   scores → τ_elbow (accept score > τ_elbow)
# Both apply to any score — the bridges here or one computed elsewhere. `buildPosterior`
# applies them via `gate` and `min_score = :elbow`. Gate types are in types/gates.jl.

"""
    openSides(plr::ProfileLikelihoodResult) -> Vector{Bool}
    openSides(lower, upper) -> Vector{Bool}

Which sides of each SM parameter's confidence interval are open, in the order
`(parameter 1 lower, parameter 1 upper, parameter 2 lower, parameter 2 upper, …)`; `true` = open.

A side is open when the profile did not cross its threshold inside the swept range, which
SmoreBase reports as a `nothing` bound (`ProfileCurve.ci_lower` / `ci_upper`). The second form
takes the bounds directly, one entry per parameter; `nothing` and `missing` both count as open.

# Example
```julia
uq = quantifyUncertainty(ProfileLikelihood(), problem, fit, 1)
openSides(uq)                      # e.g. [false, false, false, true]: parameter 2 is open above
openSides([0.1, 5.0], [0.3, nothing])   # same answer from raw bounds
```
"""
openSides(plr::ProfileLikelihoodResult) =
    openSides([pc.ci_lower for pc in plr.profiles], [pc.ci_upper for pc in plr.profiles])

function openSides(lower, upper)
    length(lower) == length(upper) || throw(DimensionMismatch(
        "lower has $(length(lower)) entries, upper has $(length(upper))"
    ))
    return Bool[x === nothing || x === missing for (l, u) in zip(lower, upper) for x in (l, u)]
end

"""
    applyGate(gate::AbstractGate, post::CMPosteriorResult) -> NamedTuple
    applyGate(gate::AbstractGate, scores, uq_results, data_profiles::ProfileLikelihoodResult)
    applyGate(gate::IdentifiabilityMatchGate, scores, open_grid::AbstractMatrix{Bool}, open_data::AbstractVector{Bool})

Apply a gate to a set of scores: every CM param_set that fails the gate has its score set to 0.
[`buildPosterior`](@ref) does this itself when given a `gate`; call `applyGate` to inspect the
gate's effect, or to gate scores computed outside this package.

# Arguments
- `gate` — an [`AbstractGate`](@ref): [`NullGate`](@ref) or [`IdentifiabilityMatchGate`](@ref)
- `post` — a [`CMPosteriorResult`](@ref); gates `post.scores` using `post.uq_results` and
  `post.data_profiles`. Gating an already gated result returns the same scores.
- `scores` — one score per CM param_set
- `uq_results` — the CM param_sets' profile results, row-aligned with `scores`
- `data_profiles` — the data's profile result
- `open_grid`, `open_data` — for [`IdentifiabilityMatchGate`](@ref) only, the CI shapes given
  directly: `[n_cm_param_sets × n_sides]` open flags (each row in [`openSides`](@ref) order)
  and the data's flags in the same order

# Returns
A named tuple with
- `scores` — the gated scores (`Vector{Float64}`)
- `pass` — `BitVector`, `true` where the CM param_set passes

and, for [`IdentifiabilityMatchGate`](@ref),
- `mismatch` — `BitMatrix`, `[n_cm_param_sets × n_sides]`: the sides on which a CM param_set's
  CI shape differs from the data's

# Example
```julia
post = buildPosterior(sm, data, uq_results, cm_params)          # raw scores
g    = applyGate(IdentifiabilityMatchGate(), post)
g.scores                  # post.scores with failing CM param_sets set to 0
count(.!g.pass)           # how many CM param_sets fail
τ    = elbowThreshold(g.scores).tau
accepted = g.scores .> τ

# or in one call:
post = buildPosterior(sm, data, uq_results, cm_params;
                      gate = IdentifiabilityMatchGate(), min_score = :elbow)
```
"""
applyGate(gate::AbstractGate, post::CMPosteriorResult) =
    applyGate(gate, post.scores, post.uq_results, post.data_profiles)

function applyGate(
    ::NullGate,
    scores::AbstractVector{<:Real},
    uq_results::AbstractVector{<:ProfileLikelihoodResult},
    ::ProfileLikelihoodResult,
)
    _checkAligned(scores, uq_results)
    return (scores = Float64.(scores), pass = trues(length(scores)))
end

function applyGate(
    gate::IdentifiabilityMatchGate,
    scores::AbstractVector{<:Real},
    uq_results::AbstractVector{<:ProfileLikelihoodResult},
    data_profiles::ProfileLikelihoodResult,
)
    _checkAligned(scores, uq_results)
    open_data = openSides(data_profiles)
    open_grid = Matrix{Bool}(undef, length(uq_results), length(open_data))
    for (i, uq) in enumerate(uq_results)
        sides = openSides(uq)
        length(sides) == length(open_data) || throw(DimensionMismatch(
            "uq_results[$i] has $(length(sides) ÷ 2) SM parameters, the data profile has $(length(open_data) ÷ 2)"
        ))
        open_grid[i, :] = sides
    end
    return applyGate(gate, scores, open_grid, open_data)
end

function applyGate(
    ::IdentifiabilityMatchGate,
    scores::AbstractVector{<:Real},
    open_grid::AbstractMatrix{Bool},
    open_data::AbstractVector{Bool},
)
    size(open_grid) == (length(scores), length(open_data)) || throw(DimensionMismatch(
        "open_grid is $(join(size(open_grid), " × ")); expected $(length(scores)) × $(length(open_data))"
    ))
    mismatch = BitMatrix(open_grid .!= reshape(open_data, 1, :))
    pass     = BitVector(.!vec(any(mismatch; dims = 2)))
    gated    = Float64[pass[i] ? scores[i] : 0.0 for i in eachindex(pass)]
    return (scores = gated, pass = pass, mismatch = mismatch)
end

_checkAligned(scores, uq_results) = length(uq_results) == length(scores) || throw(DimensionMismatch(
    "length(uq_results)=$(length(uq_results)) must equal length(scores)=$(length(scores))"
))

"""
    SmoreFit.gateInterior(gate::AbstractGate, score, lb, ub, post::CMPosteriorResult) -> Float64

Gate the `score` of one interior CM point, whose SM-parameter box `[lb, ub]` was interpolated
across the CM grid of `post`. Called by [`posteriorScore`](@ref) with `post.gate`. Add a method
for a new gate type to make interior queries work with it (see [`AbstractGate`](@ref)).

- [`NullGate`](@ref) returns `score`.
- [`IdentifiabilityMatchGate`](@ref) treats a side of the interpolated box as open when its
  bound is at (or beyond) the edge of the range the CM-side profiles swept and as bounded when
  it is strictly inside, and returns `score` if every side has the data's shape, else `0.0`.
"""
gateInterior(gate::AbstractGate, score, lb, ub, post) = throw(ArgumentError(
    "interior queries are not supported for gate $(typeof(gate)); add a method of SmoreFit.gateInterior"
))

gateInterior(::NullGate, score, lb, ub, post) = Float64(score)

function gateInterior(::IdentifiabilityMatchGate, score, lb, ub, post)
    lo, hi    = _sweptRange(post.uq_results)
    open_data = openSides(post.data_profiles)
    for j in eachindex(lb)
        (lb[j] <= lo[j]) == open_data[2j - 1] || return 0.0
        (ub[j] >= hi[j]) == open_data[2j]     || return 0.0
    end
    return Float64(score)
end

# The range the CM-side profiles swept, per SM parameter: where `_profileBox` puts an open bound.
function _sweptRange(uq_results)
    k  = length(uq_results[1].profiles)
    lo = [minimum(first(uq.profiles[j].profile_values) for uq in uq_results) for j in 1:k]
    hi = [maximum(last(uq.profiles[j].profile_values) for uq in uq_results) for j in 1:k]
    return lo, hi
end

"""
    elbowThreshold(scores) -> (tau, rank, n_positive, note)

The elbow threshold τ_elbow of a set of scores; accept a CM param_set when its score is
`> tau`.

Sort the positive scores from highest to lowest, `s[1] ≥ … ≥ s[n]`, and scale rank and score to
`[0, 1]` (`x = (rank − 1)/(n − 1)`, `y = (s − s[n])/(s[1] − s[n])`). Join the first and last
points with a straight line; the elbow is the point farthest below that line, and τ_elbow is
its score. The elbow point itself is therefore rejected, and because τ_elbow is a score value,
CM param_sets with tied scores are accepted or rejected together.

If fewer than three scores are positive, or no point lies below the line (including when all
positive scores are equal), τ_elbow is 0, which accepts every positive score.

Usually applied to gated scores (see [`applyGate`](@ref)).

# Returns
A named tuple:
- `tau` — τ_elbow
- `rank` — the elbow's rank among the sorted positive scores (0 when τ_elbow was set to 0)
- `n_positive` — the number of positive scores
- `note` — why τ_elbow was set to 0 (`""` otherwise)

# Example
```julia
g = applyGate(IdentifiabilityMatchGate(), post)
e = elbowThreshold(g.scores)
accepted = g.scores .> e.tau
```
"""
function elbowThreshold(scores::AbstractVector{<:Real})
    s = sort!(Float64[x for x in scores if x > 0]; rev = true)
    n = length(s)
    n < 3 && return (tau = 0.0, rank = 0, n_positive = n, note = "fewer than three positive scores")
    s[1] == s[n] && return (tau = 0.0, rank = 0, n_positive = n,
                            note = "no point below the line (all positive scores equal)")
    x = (0:n - 1) ./ (n - 1)
    y = (s .- s[n]) ./ (s[1] - s[n])
    below = (1 .- x) .- y                 # distance below the line, up to a constant factor
    maximum(below) > 0 || return (tau = 0.0, rank = 0, n_positive = n, note = "no point below the line")
    r = argmax(below)
    return (tau = s[r], rank = r, n_positive = n, note = "")
end
