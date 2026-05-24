# Profile the surrogate model against the real-world data, reusing the SmoreBase pipeline.
#
# `problem.data` is a single-param-set CMData (the real observations). We fit the SM to it and
# run the same profile likelihood UQ that produced the per-cohort CM-side profiles, so both
# sides live in the same SM-parameter space. `_uq` is internal to SmoreBase but documented as
# awaiting exactly this higher-level caller.
function _profileAgainstData(
    problem::SMFitProblem,
    p0,
    profile_options::ProfileLikelihood,
)
    fit = fitSurrogate(problem, p0)
    return SmoreBase._uq(problem, fit, profile_options; param_set_index = 1)
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

# Extract a cohort point's MLE SM-parameter vector from its profile likelihood result. Pulled
# from the row of `optimal_parameters` at the maximum stored log-likelihood, so this is robust
# regardless of how `uq_results` was constructed (one shared `SMFitResult` across the whole
# cohort, or an independent fit per cohort point — both yield the same MLE here).
function _cohortMLE(uq::ProfileLikelihoodResult)
    pc = first(uq.profiles)
    return pc.optimal_parameters[argmax(pc.log_likelihoods), :]
end

# Column-mean of the cohort SM fits — the documented default for the data-fit `p0`. Uses
# `_cohortMLE` per cohort so the result is genuinely a per-cohort average, not (accidentally)
# whatever shape `uq_results[1].fit_result.parameters` happens to have.
function _meanCohortMLE(uq_results::Vector{<:ProfileLikelihoodResult})
    mle_table = reduce(hcat, _cohortMLE(uq) for uq in uq_results)   # [n_sm × n_cohort]
    return reshape(mean(mle_table, dims = 2), 1, :)                 # [1   × n_sm]
end
