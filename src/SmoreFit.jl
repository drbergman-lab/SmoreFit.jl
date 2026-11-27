module SmoreFit

using SmoreBase
using Distributions
using Statistics

include("types/gates.jl")
include("types/posterior_result.jl")
include("bridge.jl")
include("data_profiles.jl")
include("posterior.jl")
include("interior.jl")
include("gate_elbow.jl")

export buildPosterior, CMPosteriorResult
export posteriorSamples, posteriorWeights, acceptedGrid, scoreGrid
export inPosterior, posteriorScore
export AbstractGate, NullGate, IdentifiabilityMatchGate
export applyGate, elbowThreshold, openSides

end
