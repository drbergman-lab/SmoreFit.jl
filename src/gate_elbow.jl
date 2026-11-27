# The two filters applied to bridge scores before a posterior is read off them:
#   gateScores       scores + a gate → which CM param_sets pass (a failing one counts as score 0)
#   elbowThreshold   scores → τ_elbow (accept score > τ_elbow)
# `buildPosterior` applies them via `gate` and `min_score = :elbow`; `applyGate` reports a
# posterior's gate result. Gate types are in types/gates.jl.

"""
    openSides(plr::ProfileLikelihoodResult) -> BitMatrix
    openSides(lower, upper) -> BitMatrix

Which sides of each SM parameter's confidence interval are open, as `[n_sm_params × 2]`: row `j`
is SM parameter `j`, column 1 its lower side, column 2 its upper side; `true` = open.

A side is open when the profile did not cross its threshold inside the prior range, which
SmoreBase reports as a `nothing` bound (`ProfileCurve.ci_lower` / `ci_upper`). The second form
takes the bounds directly, one entry per parameter; `nothing` and `missing` both count as open.

# Example
```julia
uq = quantifyUncertainty(ProfileLikelihood(), problem, fit, 1)
open = openSides(uq)     # e.g. [false false; false true]: parameter 2 is open above
open[2, 2]               # is parameter 2 open on its upper side?
any(open)                # is any side open?
openSides([0.1, 5.0], [0.3, nothing])   # same answer from raw bounds
```
"""
openSides(plr::ProfileLikelihoodResult) =
    openSides([pc.ci_lower for pc in plr.profiles], [pc.ci_upper for pc in plr.profiles])

function openSides(lower, upper)
    length(lower) == length(upper) || throw(DimensionMismatch(
        "lower has $(length(lower)) entries, upper has $(length(upper))"
    ))
    isOpen(x) = x === nothing || x === missing
    return BitMatrix([isOpen(side[j]) for j in eachindex(lower), side in (lower, upper)])
end

"""
    applyGate(post::CMPosteriorResult) -> NamedTuple
    applyGate(gate::AbstractGate, post::CMPosteriorResult) -> NamedTuple

What the gate does to a posterior's scores.

`applyGate(post)` uses the posterior's own gate, `post.gate`: the one chosen in
[`buildPosterior`](@ref) and already reflected in `post.gate_pass`, `post.accepted`,
[`scoreGrid`](@ref) and [`posteriorWeights`](@ref). Call it to see the gate's result in full.

`applyGate(gate, post)` is the special case: it evaluates a *different* gate on the same
scores, to compare gates. It returns a result and leaves `post` (and `post.gate`) unchanged; to
build a posterior with another gate, call `buildPosterior` again.

# Returns
A named tuple with
- `scores` — one score per CM param_set, 0 where it fails the gate (`Vector{Float64}`)
- `pass` — `BitVector`, `true` where the CM param_set passes

and, for [`IdentifiabilityMatchGate`](@ref),
- `mismatch` — `BitArray`, `[n_cm_param_sets × n_sm_params × 2]`: `mismatch[i, j, 1]` is `true`
  when CM param_set `i` differs from the data on the lower side of SM parameter `j`, and
  `mismatch[i, j, 2]` on the upper side (the layout of [`openSides`](@ref), one slice per CM
  param_set)

# Example
```julia
post = buildPosterior(sm, data, uq_results, cm_params; gate = IdentifiabilityMatchGate())
g    = applyGate(post)
count(.!g.pass)                   # how many CM param_sets fail the gate
dropdims(count(g.mismatch; dims = 1); dims = 1)   # failures per SM parameter (rows) and side (columns)
post.scores[post.gate_pass]   # the scores of the CM param_sets that pass

applyGate(NullGate(), post).pass  # what another gate would do: here, everything passes
```
"""
applyGate(post::CMPosteriorResult) = applyGate(post.gate, post)

applyGate(gate::AbstractGate, post::CMPosteriorResult) =
    gateScores(gate, post.scores, post.uq_results, post.data_profiles)

"""
    SmoreFit.gateScores(gate::AbstractGate, scores, uq_results, data_profiles) -> NamedTuple

Evaluate `gate` on one bridge score per CM param_set. Returns at least `scores` (0 where a CM
param_set fails) and `pass`. Not exported: this is what [`buildPosterior`](@ref) and
[`applyGate`](@ref) call, and the method to add for a new gate type (see
[`AbstractGate`](@ref)).
"""
function gateScores(
    ::NullGate,
    scores::AbstractVector{<:Real},
    uq_results::AbstractVector{<:ProfileLikelihoodResult},
    ::ProfileLikelihoodResult,
)
    _checkAligned(scores, uq_results)
    return (scores = Float64.(scores), pass = trues(length(scores)))
end

function gateScores(
    ::IdentifiabilityMatchGate,
    scores::AbstractVector{<:Real},
    uq_results::AbstractVector{<:ProfileLikelihoodResult},
    data_profiles::ProfileLikelihoodResult,
)
    _checkAligned(scores, uq_results)
    open_data = openSides(data_profiles)
    open_grid = falses(length(uq_results), size(open_data)...)
    for (i, uq) in enumerate(uq_results)
        sides = openSides(uq)
        size(sides) == size(open_data) || throw(DimensionMismatch(
            "uq_results[$i] has $(size(sides, 1)) SM parameters, the data profile has $(size(open_data, 1))"
        ))
        open_grid[i, :, :] = sides
    end
    return _matchShapes(scores, open_grid, open_data)
end

# The identifiability-match rule on open/bounded flags: `open_data` is the data's `openSides`
# matrix [n_sm_params × 2]; `open_grid` is [n_cm_param_sets × n_sm_params × 2], one such matrix
# per CM param_set.
function _matchShapes(
    scores::AbstractVector{<:Real},
    open_grid::AbstractArray{Bool,3},
    open_data::AbstractMatrix{Bool},
)
    size(open_grid) == (length(scores), size(open_data)...) || throw(DimensionMismatch(
        "open_grid is $(join(size(open_grid), " × ")); expected $(join((length(scores), size(open_data)...), " × "))"
    ))
    mismatch = BitArray(open_grid .!= reshape(open_data, 1, size(open_data)...))
    pass     = BitVector(.!vec(any(mismatch; dims = (2, 3))))
    gated    = Float64[pass[i] ? scores[i] : 0.0 for i in eachindex(pass)]
    return (scores = gated, pass = pass, mismatch = mismatch)
end

_checkAligned(scores, uq_results) = length(uq_results) == length(scores) || throw(DimensionMismatch(
    "length(uq_results)=$(length(uq_results)) must equal length(scores)=$(length(scores))"
))

"""
    SmoreFit.gateInterior(gate::AbstractGate, score, lb, ub, post::CMPosteriorResult) -> Float64

Gate the `score` of one interior CM point, whose SM-parameter box `[lb, ub]` was interpolated
across the CM grid of `post`. Not exported. Called by [`posteriorScore`](@ref) with `post.gate`.
Add a method for a new gate type to make interior queries work with it (see [`AbstractGate`](@ref)).

- [`NullGate`](@ref) returns `score`.
- [`IdentifiabilityMatchGate`](@ref) treats a side of the interpolated box as open when its
  bound is at (or beyond) the edge of the profile range and as bounded when it is strictly
  inside, and returns `score` if every side has the data's shape, else `0.0`. The profile range
  is defined in the [`IdentifiabilityMatchGate`](@ref) docstring.
"""
gateInterior(gate::AbstractGate, score, lb, ub, post) = throw(ArgumentError(
    "interior queries are not supported for gate $(typeof(gate)); add a method of SmoreFit.gateInterior"
))

gateInterior(::NullGate, score, lb, ub, post) = Float64(score)

function gateInterior(::IdentifiabilityMatchGate, score, lb, ub, post)
    lo, hi    = _profileRange(post.uq_results[1])
    open_data = openSides(post.data_profiles)
    for j in eachindex(lb)
        (lb[j] <= lo[j]) == open_data[j, 1] || return 0.0   # lower side: same shape as the data's?
        (ub[j] >= hi[j]) == open_data[j, 2] || return 0.0   # upper side
    end
    return Float64(score)
end

# The profile range of each SM parameter: the range its profile covers, known up front.
# - Normally it is the SM prior range.
# - SmoreBase cannot build a profile over an unbounded prior (for example a Normal), so
#   `ProfileLikelihood(bounds = ParameterPrior(lower, upper))` gives the profile a finite range.
#   `bounds` can also be set to narrow a finite prior range. When it is set, it is the profile range.
# An open CI bound is stored at the edge of the profile range (`_profileBox`), so that edge is
# what tells an open side from a bounded one.
function _profileRange(uq::ProfileLikelihoodResult)
    range = uq.method.bounds === nothing ? uq.fit_result.prior : uq.method.bounds
    return SmoreBase._lowerBounds(range), SmoreBase._upperBounds(range)
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

Usually applied to the scores that pass the gate, `post.scores[post.gate_pass]`;
`buildPosterior(...; min_score = :elbow)` does exactly that.

# Returns
A named tuple:
- `tau` — τ_elbow
- `rank` — the elbow's rank among the sorted positive scores (0 when τ_elbow was set to 0)
- `n_positive` — the number of positive scores
- `note` — why τ_elbow was set to 0 (`""` otherwise)

# Example
```julia
e = elbowThreshold(post.scores[post.gate_pass])
accepted = post.gate_pass .& (post.scores .> e.tau)
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
