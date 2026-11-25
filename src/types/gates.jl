"""
    AbstractGate

A gate decides, before any threshold is applied, which CM param_sets may keep their bridge
score. A CM param_set that fails the gate has its score set to 0. Pass one to
[`buildPosterior`](@ref) as `gate`, or apply one yourself with [`applyGate`](@ref).

Built-in gates: [`NullGate`](@ref) (no gating, the default) and
[`IdentifiabilityMatchGate`](@ref).

# Adding a gate
Subtype `AbstractGate` and add a method

    applyGate(gate::MyGate, scores, uq_results, data_profiles) -> NamedTuple

that returns at least `scores` (the gated scores, `Vector{Float64}`) and `pass` (a `BitVector`).
To support interior queries ([`posteriorScore`](@ref), [`inPosterior`](@ref)) also add

    SmoreFit.gateInterior(gate::MyGate, score, lb, ub, post) -> Float64

which gates the score of one interior CM point from its interpolated SM-parameter box
`[lb, ub]`. Without it, interior queries on a result built with that gate throw an
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
open where the data's is open. A CM param_set that fails has its score set to 0.

The gate uses only the CI shapes, never the scores, so it is the same for every bridge and
can be applied to scores computed outside this package.

At an interior CM point the CI shape is read off the interpolated box: a side is open if the
interpolated bound sits at the edge of the range the CM-side profiles swept, and bounded if it
lies strictly inside. So a point interpolated only from CM param_sets that are open on a side
is open there, and one interpolated only from bounded CM param_sets is bounded.
"""
struct IdentifiabilityMatchGate <: AbstractGate end
