using SmoreFit
using SmoreBase
using Statistics
using Test

# Logistic growth K / (1 + (K/y0 - 1) exp(-r t)); SM params p = [r, K], y0 = 0.01.
_logistic(t, p, _c) = reshape(
    p[2] ./ (1.0 .+ (p[2] / 0.01 - 1.0) .* exp.(-p[1] .* t)),
    :, 1,
)

@testset "buildPosterior" begin
    sm       = AnalyticalSurrogateModel(fn = _logistic)
    t        = collect(0.0:5.0:50.0)                      # saturates → r and K identifiable
    sm_prior = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])
    P0       = [0.5 5.0]
    noise    = 0.02

    # CM grid of 5 cohort points; CM parameter c sets the true SM growth rate r.
    cvals    = [1.0, 2.0, 3.0, 4.0, 5.0]
    rtrue(c) = 0.35 + 0.05 * c                            # r ∈ {0.40, …, 0.60}
    Ktrue    = 4.0

    function _profileAt(c)
        μ    = _logistic(t, [rtrue(c), Ktrue], nothing)
        d    = CMData(μ = vec(μ), σ = noise .* ones(length(t)), times = t)
        prob = SMFitProblem(sm, d, sm_prior)
        fit  = fitSurrogate(prob, P0)
        return SmoreBase._uq(prob, fit, ProfileLikelihood(n_points = 12))
    end

    uq_results = [_profileAt(c) for c in cvals]
    cm_params  = reshape(cvals, :, 1)                     # [5 × 1], a regular grid
    cm_prior   = ParameterPrior([1.0], [5.0]; names = ["c"])

    # Real data generated at the center cohort (c = 3 → r = 0.50).
    μ_data = _logistic(t, [rtrue(3.0), Ktrue], nothing)
    data   = CMData(μ = vec(μ_data), σ = noise .* ones(length(t)), times = t)

    @testset "bridge = $bridge" for bridge in (:box_overlap, :data_trace_in_box, :symmetric_trace)
        post = buildPosterior(sm, data, uq_results, cm_params, cm_prior; bridge = bridge)
        @test post isa CMPosteriorResult
        @test length(post.scores) == 5
        @test post.bridge == bridge

        # The generating (center) cohort is accepted and scores highest.
        @test post.accepted[3]
        @test argmax(post.scores) == 3

        # Far cohorts have disjoint SM regions → rejected at the default tolerance.
        @test !post.accepted[1]
        @test !post.accepted[5]

        # Scores decrease moving away from the center on both sides.
        @test post.scores[3] >= post.scores[2] >= post.scores[1]
        @test post.scores[3] >= post.scores[4] >= post.scores[5]

        # All scores are valid consistency values in [0, 1].
        @test all(0.0 .<= post.scores .<= 1.0)
    end

    @testset "posterior outputs" begin
        post    = buildPosterior(sm, data, uq_results, cm_params, cm_prior)
        samples = posteriorSamples(post)
        @test size(samples, 2) == 1
        @test size(samples, 1) == count(post.accepted)
        @test all(in(cvals), samples[:, 1])

        # Grid-aware reshaping (1-D grid → length-5, axis sorted to match input order).
        @test length(acceptedGrid(post)) == 5
        @test acceptedGrid(post)[3] == true
        @test scoreGrid(post) == post.scores

        # data_profiles is the SM profiled against the real data.
        @test post.data_profiles isa ProfileLikelihoodResult
        @test length(post.data_profiles.profiles) == 2
    end

    @testset "graded posterior" begin
        post = buildPosterior(sm, data, uq_results, cm_params, cm_prior; posterior = :graded)
        @test post.posterior == :graded
        w = posteriorWeights(post)
        @test isapprox(sum(w), 1.0; atol = 1e-12)
        @test argmax(w) == 3
    end

    @testset "SMFitProblem dispatch" begin
        # If the user already has an SMFitProblem in hand, it should drive the call directly
        # and produce the same posterior as the (sm, data) form.
        sm_prior = uq_results[1].fit_result.prior
        problem  = SMFitProblem(sm, data, sm_prior)

        post_sm      = buildPosterior(sm, data, uq_results, cm_params, cm_prior)
        post_problem = buildPosterior(problem, uq_results, cm_params, cm_prior)
        @test post_problem.accepted == post_sm.accepted
        @test post_problem.scores   ≈  post_sm.scores

        # Same path with an explicit CMSample.
        post_problem_sample = buildPosterior(problem, uq_results, CMSample(cm_params), cm_prior)
        @test post_problem_sample.accepted == post_sm.accepted

        # Validation propagates from the problem's data.
        bad     = CMData(μ = rand(length(t), 2), σ = ones(length(t), 2), times = t, param_sets = 2)
        bad_prob = SMFitProblem(sm, bad, sm_prior)
        @test_throws ArgumentError buildPosterior(bad_prob, uq_results, cm_params, cm_prior)
    end

    @testset "default p0 = mean of per-cohort MLEs" begin
        # Regression: the default `p0` must be the cohort column-mean of *per-cohort* MLEs,
        # not whatever `uq_results[1].fit_result.parameters` happens to contain. To catch the
        # failure mode where the default would silently collapse onto cohort point 1, we feed
        # `buildPosterior` a `uq_results` constructed so that each entry's `fit_result` carries
        # only its own row — and verify the data-fit init still averages across the cohort.

        per_point_uq = map(uq -> let
            # Reduce fit_result.parameters to a single-row table containing only this cohort's
            # MLE — mimics the user who fit each cohort point independently.
            mle = SmoreFit._cohortMLE(uq)
            fit = SMFitResult(
                reshape(mle, 1, :),
                [uq.fit_result.errors[1]],
                reshape(mle, 1, :),
                uq.fit_result.prior,
                BitVector([true]),
                Any[nothing],
            )
            ProfileLikelihoodResult{Float64}(uq.profiles, fit, uq.times)
        end, uq_results)

        @test size(per_point_uq[1].fit_result.parameters, 1) == 1   # confirm the setup

        # The default p0 from this per-cohort `uq_results` must equal the cohort column-mean.
        cohort_mles = reduce(hcat, [SmoreFit._cohortMLE(uq) for uq in uq_results])
        expected    = reshape(Statistics.mean(cohort_mles, dims = 2), 1, :)
        @test SmoreFit._meanCohortMLE(per_point_uq) ≈ expected

        # And `buildPosterior` produces the same posterior as the shared-fit form — i.e. the
        # default p0 doesn't silently degrade with the per-point construction.
        post_shared    = buildPosterior(sm, data, uq_results,     cm_params, cm_prior)
        post_per_point = buildPosterior(sm, data, per_point_uq,   cm_params, cm_prior)
        @test post_per_point.accepted == post_shared.accepted
        @test post_per_point.scores   ≈  post_shared.scores
    end

    @testset "interior queries" begin
        post = buildPosterior(sm, data, uq_results, cm_params, cm_prior)   # :box_overlap

        # Reproduce stored scores/acceptance at the cohort points themselves.
        for i in 1:5
            @test posteriorScore(post, [cvals[i]]) ≈ post.scores[i]
            @test inPosterior(post, [cvals[i]])    == post.accepted[i]
        end

        # An interior CM point (off the cohort grid) that is also far from c = 3 — the
        # cohort point that generated the data — is rejected. We don't assert acceptance of a
        # *near* interior point: the data noise is narrow enough that the accepted window
        # around c = 3 is much smaller than the cohort step, so any "close to c = 3" test
        # would be brittle to grid spacing and noise. The robust half of the picture — far
        # from the data-generating point in CM space → score 0 — is what's tested here.
        @test posteriorScore(post, [1.5]) == 0.0    # c = 1.5: between cohort c = 1 and c = 2, far from c = 3
        @test !inPosterior(post, [1.5])

        # Batch form agrees with point-by-point.
        queries = reshape([1.5, 2.5, 3.5, 4.5], :, 1)
        s_batch = posteriorScore(post, queries)
        s_loop  = [posteriorScore(post, queries[k, :]) for k in 1:size(queries, 1)]
        @test s_batch ≈ s_loop
        @test inPosterior(post, queries) == BitVector(s_loop .> post.acceptance_tol)

        # tol override widens or tightens the threshold.
        widest = inPosterior(post, [1.5]; tol = -1.0)
        @test widest == true                # any score > -1 ⇒ accepted
        @test inPosterior(post, [3.0]; tol = 2.0) == false  # impossible threshold

        # :symmetric_trace and ScatteredCMSample → clear errors.
        post_sym = buildPosterior(sm, data, uq_results, cm_params, cm_prior; bridge = :symmetric_trace)
        @test_throws ArgumentError posteriorScore(post_sym, [3.0])
        @test_throws ArgumentError inPosterior(post_sym, [3.0])

        scattered = ScatteredCMSample(cm_params)
        post_sc = buildPosterior(sm, data, uq_results, scattered, cm_prior)
        @test post_sc.get_bounds === nothing
        @test_throws ArgumentError posteriorScore(post_sc, [3.0])
    end

    @testset "_boxOverlapScore degenerate data dimensions" begin
        # Regression: a zero-width data dimension was incorrectly short-circuiting the score
        # to 0 even when the CM box contained the data point. The docstring promises factor 1
        # in that case, so a containing CM box should still let other dimensions decide the
        # final score.

        # 2-D: data dim 1 is degenerate at 0.5 (CM [0, 1] contains it → factor 1);
        # dim 2 has CM == data == [0.0, 0.7] → factor 1. Total: 1.0, not 0.0.
        @test SmoreFit._boxOverlapScore([0.0, 0.0], [1.0, 0.7], [0.5, 0.0], [0.5, 0.7]) ≈ 1.0

        # Degenerate dim contributes factor 1 only when contained; otherwise 0.
        @test SmoreFit._boxOverlapScore([0.6, 0.0], [1.0, 1.0], [0.5, 0.5], [0.5, 0.7]) == 0.0

        # Boundary of the CM box still counts as containing the degenerate data point.
        @test SmoreFit._boxOverlapScore([0.5, 0.0], [1.0, 1.0], [0.5, 0.5], [0.5, 0.7]) > 0.0

        # Non-degenerate disjoint dimension still short-circuits to 0.
        @test SmoreFit._boxOverlapScore([0.0, 0.8], [1.0, 1.0], [0.5, 0.0], [0.5, 0.7]) == 0.0

        # All-degenerate matching point: every dim contributes 1 → score 1.0.
        @test SmoreFit._boxOverlapScore([0.0, 0.0], [1.0, 1.0], [0.5, 0.3], [0.5, 0.3]) ≈ 1.0
    end

    @testset "validation" begin
        # data must have a single param set
        bad = CMData(μ = rand(length(t), 2), σ = ones(length(t), 2), times = t, param_sets = 2)
        @test_throws ArgumentError buildPosterior(sm, bad, uq_results, cm_params, cm_prior)

        # uq_results length must match the number of cohort points
        @test_throws ArgumentError buildPosterior(sm, data, uq_results[1:3], cm_params, cm_prior)

        # unknown bridge method
        @test_throws ArgumentError buildPosterior(sm, data, uq_results, cm_params, cm_prior; bridge = :nope)

        # unknown posterior mode
        @test_throws ArgumentError buildPosterior(sm, data, uq_results, cm_params, cm_prior; posterior = :nope)
    end
end
