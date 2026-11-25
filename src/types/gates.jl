"""
    AbstractGate

A gate decides, before any threshold is applied, which CM param_sets may count at all. A CM
param_set that fails the gate is treated as score 0 and is never accepted.

A posterior has one gate. It is chosen in [`buildPosterior`](@ref) (`gate = ...`), stored as
`post.gate`, and used by everything that reads the posterior afterwards. [`applyGate`](@ref)
reports what it did.

Built-in gates: [`NullGate`](@ref) (no gating, the default) and
[`IdentifiabilityMatchGate`](@ref).

# Adding a gate
Subtype `AbstractGate` and add a method of the (unexported) function

    SmoreFit.gateScores(gate::MyGate, scores, uq_results, data_profiles) -> NamedTuple

that returns at least `scores` (the scores with 0 where a CM param_set fails, `Vector{Float64}`)
and `pass` (a `BitVector`). `scores` holds one bridge score per CM param_set, `uq_results` the
CM param_sets' profile results in the same order, and `data_profiles` the data's profile result.
To support interior queries ([`posteriorScore`](@ref), [`inPosterior`](@ref)) also add

    SmoreFit.gateInterior(gate::MyGate, score, lb, ub, post) -> Float64

which gates the score of one interior CM point from its interpolated SM-parameter box
`[lb, ub]`. Without it, interior queries on a posterior built with that gate throw an
`ArgumentError`.
"""
abstract type AbstractGate end

"""
    NullGate()

The gate that lets every CM param_set through: scores are left as the bridge computed them.
The default `gate` of [`buildPosterior`](@ref).
"""
struct NullGate <: AbstractGate end

"""
    IdentifiabilityMatchGate()

The identifiability gate. Each side of each SM parameter's confidence interval is either
bounded or open (see [`openSides`](@ref)). A CM param_set passes if, on every side of every SM
parameter, its CI has the same shape as the data's CI: bounded where the data's is bounded,
open where the data's is open. A CM param_set that fails counts as score 0.

The gate uses only the CI shapes, never the scores, so it is the same for every bridge.

At an interior CM point the CI shape is read off the interpolated box: a side is open if the
interpolated bound sits at the edge of the SM prior range, and bounded if it
lies strictly inside. So a point interpolated only from CM param_sets that are open on a side
is open there, and one interpolated only from bounded CM param_sets is bounded.
"""
struct IdentifiabilityMatchGate <: AbstractGate end
