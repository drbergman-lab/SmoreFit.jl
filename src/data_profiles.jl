# Profile the surrogate model against the real-world data, reusing the SmoreBase pipeline.
#
# `problem.data` is a single-param-set CMData (the real observations). We fit the SM to it and
# run the same profile likelihood UQ that produced the per-cm_param_set CM-side profiles, so both
# sides live in the same SM-parameter space.
function _profileAgainstData(
    problem::SMFitProblem,
    p0,
    profile_options::ProfileLikelihood,
)
    fit = fitSurrogate(problem, p0)
    return quantifyUncertainty(profile_options, problem, fit, 1)
end

# Convenience overload: build the SMFitProblem from its pieces and delegate.
function _profileAgainstData(
    sm::AbstractSurrogateModel,
    data::AbstractCMData,
    sm_prior::ParameterPrior,
    loss::AbstractLoss,
    p0,
    profile_options::ProfileLikelihood,
)
    return _profileAgainstData(
        SMFitProblem(sm, data, sm_prior; loss = loss),
        p0,
        profile_options,
    )
end

# Extract a cm_param_set's MLE SM-parameter vector from its profile likelihood result. Pulled
# from the row of `optimal_parameters` at the maximum stored log-likelihood, so this is robust
# regardless of how `uq_results` was constructed (one shared `SMFitResult` across all CM param_sets,
# or an independent fit per cm_param_set — both yield the same MLE here).
function _cmParamSetMLE(uq::ProfileLikelihoodResult)
    pc = first(uq.profiles)
    return pc.optimal_parameters[argmax(pc.log_likelihoods), :]
end

# Column-mean of the CM param_sets' SM fits — the documented default for the data-fit `p0`. Uses
# `_cmParamSetMLE` per param_set so the result is genuinely a per-cm_param_set average, not (accidentally)
# whatever shape `uq_results[1].fit_result.parameters` happens to have.
function _meanCmParamSetMLE(uq_results::Vector{<:ProfileLikelihoodResult})
    mle_table = reduce(hcat, _cmParamSetMLE(uq) for uq in uq_results)   # [n_sm × n_cm_param_sets]
    return reshape(mean(mle_table, dims = 2), 1, :)                 # [1   × n_sm]
end
