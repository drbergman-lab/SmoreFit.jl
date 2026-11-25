using SmoreFit
using SmoreBase
using Statistics
using Test

# Logistic growth K / (1 + (K/y0 - 1) exp(-r t)); SM params p = [r, K], y0 = 0.01.
_logistic(t, p, _c) = reshape(
    p[2] ./ (1.0 .+ (p[2] / 0.01 - 1.0) .* exp.(-p[1] .* t)),
    :, 1,
)

# Open/bounded flags written one row per CM param_set as (p1 lower, p1 upper, p2 lower, p2 upper, …),
# rearranged to [n_cm_param_sets × n_sm_params × 2] (the layout of `openSides`, per CM param_set).
_sideArray(flags) = cat(flags[:, 1:2:end], flags[:, 2:2:end]; dims = 3)

# A user-defined gate: only the first CM param_set keeps its score.
struct _KeepFirstGate <: AbstractGate end
SmoreFit.gateScores(::_KeepFirstGate, scores, uq_results, data_profiles) =
    (scores = [i == 1 ? Float64(scores[i]) : 0.0 for i in eachindex(scores)], pass = BitVector(eachindex(scores) .== 1))

@testset "buildPosterior" begin
    sm       = CustomSurrogateModel(fn = _logistic)
    t        = collect(0.0:5.0:50.0)                      # saturates → r and K identifiable
    sm_prior = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])
    P0       = [0.5 5.0]
    noise    = 0.02

    # CM grid of 5 cm_param_sets; CM parameter c sets the true SM growth rate r.
    cvals    = [1.0, 2.0, 3.0, 4.0, 5.0]
    rtrue(c) = 0.35 + 0.05 * c                            # r ∈ {0.40, …, 0.60}
    Ktrue    = 4.0

    function _profileAt(c; σ = noise)
        μ    = _logistic(t, [rtrue(c), Ktrue], nothing)
        d    = CMData(μ = vec(μ), σ = σ .* ones(length(t)), times = t)
        prob = SMFitProblem(sm, d, sm_prior)
        fit  = fitSurrogate(prob, P0)
        return quantifyUncertainty(ProfileLikelihood(n_points = 12), prob, fit, 1)
    end

    uq_results = [_profileAt(c) for c in cvals]
    cm_params  = reshape(cvals, :, 1)                     # [5 × 1], a regular grid

    # Real data generated at the center CM param_set (c = 3 → r = 0.50).
    μ_data = _logistic(t, [rtrue(3.0), Ktrue], nothing)
    data   = CMData(μ = vec(μ_data), σ = noise .* ones(length(t)), times = t)

    @testset "bridge = $bridge" for bridge in (:box_overlap, :data_trace_in_box, :symmetric_trace)
        post = buildPosterior(sm, data, uq_results, cm_params; bridge = bridge)
        @test post isa CMPosteriorResult
        @test length(post.raw_scores) == 5
        @test post.bridge == bridge

        # The generating (center) CM param_set is accepted and scores highest.
        @test post.accepted[3]
        @test argmax(post.raw_scores) == 3

        # Far CM param_sets have disjoint SM regions → rejected at the default tolerance.
        @test !post.accepted[1]
        @test !post.accepted[5]

        # Scores decrease moving away from the center on both sides.
        @test post.raw_scores[3] >= post.raw_scores[2] >= post.raw_scores[1]
        @test post.raw_scores[3] >= post.raw_scores[4] >= post.raw_scores[5]

        # All scores are valid consistency values in [0, 1].
        @test all(0.0 .<= post.raw_scores .<= 1.0)
    end

    @testset "posterior outputs" begin
        post    = buildPosterior(sm, data, uq_results, cm_params)
        samples = posteriorSamples(post)
        @test size(samples, 2) == 1
        @test size(samples, 1) == count(post.accepted)
        @test all(in(cvals), samples[:, 1])

        # Grid-aware reshaping (1-D grid → length-5, axis sorted to match input order).
        @test length(acceptedGrid(post)) == 5
        @test acceptedGrid(post)[3] == true
        @test scoreGrid(post) == post.raw_scores

        # data_profiles is the SM profiled against the real data.
        @test post.data_profiles isa ProfileLikelihoodResult
        @test length(post.data_profiles.profiles) == 2
    end

    @testset "cm_names" begin
        # Raw matrix, no names given -> CMSample's auto-generated default.
        post_default = buildPosterior(sm, data, uq_results, cm_params)
        @test post_default.cm_names == ["cm_1"]

        # cm_sample already carries names -> used as the default.
        cm_sample_named = GridCMSample(cm_params; names = ["c"])
        post_named = buildPosterior(sm, data, uq_results, cm_sample_named)
        @test post_named.cm_names == ["c"]

        # Explicit cm_names kwarg overrides the cm_sample default.
        post_override = buildPosterior(sm, data, uq_results, cm_sample_named; cm_names = ["cm_override"])
        @test post_override.cm_names == ["cm_override"]

        # Explicit cm_names kwarg on the raw-matrix form threads through to the built CMSample.
        post_matrix_named = buildPosterior(sm, data, uq_results, cm_params; cm_names = ["c"])
        @test post_matrix_named.cm_names == ["c"]

        # A cm_names length mismatched against the number of CM parameters must throw cleanly,
        # not silently mislabel the posterior.
        @test_throws ArgumentError buildPosterior(sm, data, uq_results, cm_sample_named; cm_names = ["too", "many"])
        @test_throws ArgumentError buildPosterior(sm, data, uq_results, cm_params; cm_names = String[])
    end

    @testset "graded posterior" begin
        post = buildPosterior(sm, data, uq_results, cm_params)
        w = posteriorWeights(post)
        @test isapprox(sum(w), 1.0; atol = 1e-12)
        @test argmax(w) == 3
    end

    @testset "SMFitProblem dispatch" begin
        # If the user already has an SMFitProblem in hand, it should drive the call directly
        # and produce the same posterior as the (sm, data) form.
        sm_prior = uq_results[1].fit_result.prior
        problem  = SMFitProblem(sm, data, sm_prior)

        post_sm      = buildPosterior(sm, data, uq_results, cm_params)
        post_problem = buildPosterior(problem, uq_results, cm_params)
        @test post_problem.accepted == post_sm.accepted
        @test post_problem.raw_scores   ≈  post_sm.raw_scores

        # Same path with an explicit CMSample.
        post_problem_sample = buildPosterior(problem, uq_results, CMSample(cm_params))
        @test post_problem_sample.accepted == post_sm.accepted

        # The data-side problem is recorded: the one passed, or the one the (sm, data) form built.
        @test post_problem.data_problem === problem
        @test post_problem_sample.data_problem === problem
        @test post_sm.data_problem isa SMFitProblem
        @test post_sm.data_problem.sm === sm
        @test post_sm.data_problem.data === data
        @test post_sm.data_problem.prior === sm_prior
        @test post_sm.data_problem.loss isa GaussianNLL
        custom = CustomLoss((A, d, ki) -> SmoreBase._computeLoss(GaussianNLL(), A, d, ki))
        @test buildPosterior(sm, data, uq_results, CMSample(cm_params); loss = custom).data_problem.loss === custom

        # Validation propagates from the problem's data.
        bad     = CMData(μ = rand(length(t), 2), σ = ones(length(t), 2), times = t, cm_param_sets = 2)
        bad_prob = SMFitProblem(sm, bad, sm_prior)
        @test_throws ArgumentError buildPosterior(bad_prob, uq_results, cm_params)
    end

    @testset "default p0 = mean of per-cm_param_set MLEs" begin
        # Regression: the default `p0` must be the CM param_sets' column-mean of *per-cm_param_set* MLEs,
        # not whatever `uq_results[1].fit_result.parameters` happens to contain. To catch the
        # failure mode where the default would silently collapse onto cm_param_set 1, we feed
        # `buildPosterior` a `uq_results` constructed so that each entry's `fit_result` carries
        # only its own row — and verify the data-fit init still averages across the CM param_sets.

        per_point_uq = map(uq -> let
            # Reduce fit_result.parameters to a single-row table containing only this cm_param_set's
            # MLE — mimics the user who fit each cm_param_set independently.
            mle = SmoreFit._cmParamSetMLE(uq)
            fit = SMFitResult(
                reshape(mle, 1, :),
                [uq.fit_result.errors[1]],
                reshape(mle, 1, :),
                uq.fit_result.prior,
                BitVector([true]),
                Any[nothing],
            )
            ProfileLikelihoodResult{Float64}(uq.profiles, fit, uq.times, uq.method)
        end, uq_results)

        @test size(per_point_uq[1].fit_result.parameters, 1) == 1   # confirm the setup

        # The default p0 from this per-cm_param_set `uq_results` must equal the CM param_sets' column-mean.
        cm_param_set_mles = reduce(hcat, [SmoreFit._cmParamSetMLE(uq) for uq in uq_results])
        expected    = reshape(Statistics.mean(cm_param_set_mles, dims = 2), 1, :)
        @test SmoreFit._meanCmParamSetMLE(per_point_uq) ≈ expected

        # And `buildPosterior` produces the same posterior as the shared-fit form — i.e. the
        # default p0 doesn't silently degrade with the per-point construction.
        post_shared    = buildPosterior(sm, data, uq_results,     cm_params)
        post_per_point = buildPosterior(sm, data, per_point_uq,   cm_params)
        @test post_per_point.accepted == post_shared.accepted
        @test post_per_point.raw_scores   ≈  post_shared.raw_scores
    end

    @testset "interior queries" begin
        post = buildPosterior(sm, data, uq_results, cm_params)   # :box_overlap

        # Reproduce stored scores/acceptance at the cm_param_sets themselves.
        for i in 1:5
            @test posteriorScore(post, [cvals[i]]) ≈ post.raw_scores[i]
            @test inPosterior(post, [cvals[i]])    == post.accepted[i]
        end

        # An interior CM point (off the CM param_set grid) that is also far from c = 3 — the
        # cm_param_set that generated the data — is rejected. We don't assert acceptance of a
        # *near* interior point: the data noise is narrow enough that the accepted window
        # around c = 3 is much smaller than the CM param_set step, so any "close to c = 3" test
        # would be brittle to grid spacing and noise. The robust half of the picture — far
        # from the data-generating point in CM space → score 0 — is what's tested here.
        @test posteriorScore(post, [1.5]) == 0.0    # c = 1.5: between CM param_sets c = 1 and c = 2, far from c = 3
        @test !inPosterior(post, [1.5])

        # Batch form agrees with point-by-point.
        queries = reshape([1.5, 2.5, 3.5, 4.5], :, 1)
        s_batch = posteriorScore(post, queries)
        s_loop  = [posteriorScore(post, queries[k, :]) for k in 1:size(queries, 1)]
        @test s_batch ≈ s_loop
        @test inPosterior(post, queries) == BitVector(s_loop .> post.min_score)

        # min_score override widens or tightens the threshold.
        widest = inPosterior(post, [1.5]; min_score = -1.0)
        @test widest == true                # any score > -1 ⇒ accepted
        @test inPosterior(post, [3.0]; min_score = 2.0) == false  # impossible threshold

        # :symmetric_trace and ScatteredCMSample → clear errors.
        post_sym = buildPosterior(sm, data, uq_results, cm_params; bridge = :symmetric_trace)
        @test_throws ArgumentError posteriorScore(post_sym, [3.0])
        @test_throws ArgumentError inPosterior(post_sym, [3.0])

        scattered = ScatteredCMSample(cm_params)
        post_sc = buildPosterior(sm, data, uq_results, scattered)
        @test post_sc.get_bounds === nothing
        @test post_sc.interp isa LinearCIInterp
        @test_throws ArgumentError posteriorScore(post_sc, [3.0])
    end

    @testset "provenance" begin
        opts = ProfileLikelihood(n_points = 9)
        post = buildPosterior(sm, data, uq_results, cm_params; profile_options = opts,
                              interp = LinearCIInterp())

        # Both sides' profile settings: data side from profile_options, CM side from upstream.
        @test post.data_profiles.method === opts
        @test all(uq -> uq.method == ProfileLikelihood(n_points = 12), post.uq_results)
        @test length(post.data_profiles.profiles[1].profile_values) == 9

        # get_bounds is a named callable and can be rebuilt from the stored fields alone.
        @test post.interp isa LinearCIInterp
        @test post.get_bounds isa SmoreBase.CIBoundsInterpolant
        rebuilt = SmoreBase._buildBoundsInterpolant(post.cm_sample, post.lb_table, post.ub_table, post.interp)
        for θ in ([1.0], [2.5], [3.7], [5.0])
            @test rebuilt(θ) == post.get_bounds(θ)
        end
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

    @testset "box scores" begin
        J  = SmoreFit._boxJaccardScore
        DN = SmoreFit._boxOverlapScore

        @testset "Jaccard: hand-computed values" begin
            # 1-D
            @test J([0.0], [2.0], [1.0], [3.0]) ≈ 1 / 3     # partial: overlap 1, union 3
            @test J([0.0], [4.0], [1.0], [2.0]) ≈ 1 / 4     # CM box contains data box: smaller ÷ larger
            @test J([1.0], [2.0], [0.0], [4.0]) ≈ 1 / 4     # data box contains CM box
            @test J([0.0], [1.0], [0.0], [1.0]) == 1.0      # identical
            @test J([0.0], [1.0], [2.0], [3.0]) == 0.0      # disjoint
            @test J([0.0], [1.0], [1.0], [2.0]) == 0.0      # touching: zero-measure overlap

            # 2-D
            @test J([0.0, 0.0], [2.0, 2.0], [1.0, 1.0], [3.0, 3.0]) ≈ 1 / 7    # overlap 1, union 4 + 4 - 1
            @test J([0.0, 0.0], [4.0, 4.0], [1.0, 1.0], [2.0, 3.0]) ≈ 2 / 16   # containment: 2 ÷ 16
            @test J([0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [2.0, 3.0]) == 1.0     # identical
            @test J([0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [1.0, 3.0]) == 0.0     # disjoint in dim 2 only
        end

        @testset "Jaccard is symmetric; data-normalized is not" begin
            a_lo, a_hi = [0.0, 0.0], [2.0, 1.0]   # volume 2
            b_lo, b_hi = [1.0, 0.5], [4.0, 2.0]   # volume 4.5; overlap [1, 2] × [0.5, 1] = 0.5
            @test J(a_lo, a_hi, b_lo, b_hi) ≈ 0.5 / (2 + 4.5 - 0.5)
            @test J(a_lo, a_hi, b_lo, b_hi) == J(b_lo, b_hi, a_lo, a_hi)
            @test DN(a_lo, a_hi, b_lo, b_hi) ≈ 0.5 / 4.5
            @test DN(b_lo, b_hi, a_lo, a_hi) ≈ 0.5 / 2
        end

        @testset "open CI: wide CM box containing the data box" begin
            # An open profile CI stretches the CM box to the swept-range edge. The data-normalized
            # score gives such a box the maximum score however wide it is; Jaccard penalizes it.
            cm_lo, cm_hi = [0.0, 0.0], [10.0, 10.0]
            d_lo,  d_hi  = [0.9, 1.9], [1.1, 2.1]
            @test DN(cm_lo, cm_hi, d_lo, d_hi) == 1.0
            @test J(cm_lo, cm_hi, d_lo, d_hi) ≈ 0.04 / 100
        end

        @testset "Jaccard: no underflow or overflow at extreme scales" begin
            # Multiplying raw widths underflows (1e-9^40) or overflows (1e9^40) to 0 ÷ 0 = NaN.
            for w in (1e-9, 1e9)
                lo, hi = zeros(40), fill(w, 40)
                @test J(lo, hi, lo, hi) == 1.0                                   # identical
                @test J(lo, 2 .* hi, hi, 3 .* hi) ≈ 1 / (2 * 2.0^40 - 1)          # half overlap per dim
            end
            # A volume ratio that overflows gives 0, not NaN.
            @test J(zeros(400), fill(10.0, 400), zeros(400), ones(400)) == 0.0
        end

        @testset "Jaccard: degenerate dimensions" begin
            # A zero-width dimension in either box gives a zero-volume overlap → 0, symmetrically,
            # even when both boxes are zero-width at the same point (0 ÷ 0 → 0, never NaN).
            cases = [
                # (cm_lo, cm_hi, d_lo, d_hi): the `_boxOverlapScore` regression cases, then
                # zero-width CM dimensions
                ([0.0, 0.0], [1.0, 0.7], [0.5, 0.0], [0.5, 0.7]),   # data point inside CM box
                ([0.6, 0.0], [1.0, 1.0], [0.5, 0.5], [0.5, 0.7]),   # data point outside CM box
                ([0.5, 0.0], [1.0, 1.0], [0.5, 0.5], [0.5, 0.7]),   # data point on CM boundary
                ([0.0, 0.8], [1.0, 1.0], [0.5, 0.0], [0.5, 0.7]),   # plus a disjoint dimension
                ([0.0, 0.0], [1.0, 1.0], [0.5, 0.3], [0.5, 0.3]),   # data box a single point
                ([0.5, 0.0], [0.5, 1.0], [0.0, 0.0], [1.0, 1.0]),   # zero-width CM dim
                ([0.5, 0.0], [0.5, 1.0], [0.5, 0.0], [0.5, 1.0]),   # both zero-width at the same point
            ]
            for (cm_lo, cm_hi, d_lo, d_hi) in cases
                @test J(cm_lo, cm_hi, d_lo, d_hi) === 0.0
                @test J(d_lo, d_hi, cm_lo, cm_hi) === 0.0
            end
        end

        @testset "invalid box_score" begin
            @test_throws ArgumentError SmoreFit._boxScore([0.0], [1.0], [0.0], [1.0], :union)
        end
    end

    @testset "buildPosterior box_score" begin
        post_dn = buildPosterior(sm, data, uq_results, cm_params; box_score = :data_normalized)
        post_j  = buildPosterior(sm, data, uq_results, cm_params; box_score = :jaccard)
        @test post_dn.box_score == :data_normalized
        @test post_j.box_score  == :jaccard

        # Every CM box here lies inside the data box (the CM-side profiles are tighter), and then
        # union = data box, so the two scores coincide.
        @test post_j.raw_scores ≈ post_dn.raw_scores
        @test post_j.accepted == post_dn.accepted

        # A CM box containing the data box separates them: noisier CM output at c = 3 widens its
        # SM region past the data's. Data-normalized gives it 1.0; Jaccard gives data-box volume
        # ÷ CM-box volume.
        uq_wide    = copy(uq_results)
        uq_wide[3] = _profileAt(3.0; σ = 5noise)
        wide_dn = buildPosterior(sm, data, uq_wide, cm_params; box_score = :data_normalized)
        wide_j  = buildPosterior(sm, data, uq_wide, cm_params; box_score = :jaccard)
        cm_lo, cm_hi = SmoreFit._profileBox(uq_wide[3])
        d_lo,  d_hi  = SmoreFit._profileBox(wide_j.data_profiles)
        @test all(cm_lo .<= d_lo) && all(d_hi .<= cm_hi)   # confirm the setup: containment
        @test wide_dn.raw_scores[3] ≈ 1.0
        @test wide_j.raw_scores[3]  ≈ prod(d_hi .- d_lo) / prod(cm_hi .- cm_lo)
        @test wide_j.raw_scores[3]  < 0.1

        # Overlap > 0 under either score, so acceptance at min_score = 0 is the same.
        @test wide_j.accepted == wide_dn.accepted
        @test iszero.(wide_j.raw_scores) == iszero.(wide_dn.raw_scores)

        # Interior queries reuse the stored box_score, reproducing the stored scores.
        for post in (post_dn, post_j, wide_dn, wide_j), i in 1:5
            @test posteriorScore(post, [cvals[i]]) ≈ post.raw_scores[i]
        end

        # Ignored by the trace bridges, without error.
        for bridge in (:data_trace_in_box, :symmetric_trace)
            a = buildPosterior(sm, data, uq_results, cm_params; bridge = bridge, box_score = :jaccard)
            b = buildPosterior(sm, data, uq_results, cm_params; bridge = bridge, box_score = :data_normalized)
            @test a.raw_scores == b.raw_scores
        end
    end

    @testset "box_score default (v0.6): Jaccard, no warning" begin
        post_default = @test_logs min_level = Base.CoreLogging.Warn buildPosterior(sm, data, uq_results, cm_params)
        post_j       = buildPosterior(sm, data, uq_results, cm_params; box_score = :jaccard)
        @test post_default.box_score == :jaccard
        @test post_default.raw_scores   == post_j.raw_scores
        @test post_default.accepted == post_j.accepted

        # Where the scores differ (a CM box containing the data box), the default is Jaccard's.
        uq_wide    = copy(uq_results)
        uq_wide[3] = _profileAt(3.0; σ = 5noise)
        wide_default = buildPosterior(sm, data, uq_wide, cm_params)
        wide_j       = buildPosterior(sm, data, uq_wide, cm_params; box_score = :jaccard)
        @test wide_default.raw_scores == wide_j.raw_scores
        @test wide_default.raw_scores[3] < 0.1
    end

    @testset "zero-width CI scores 0 under Jaccard" begin
        # Collapse the K profile's CI of cm_param_set 2 to a point.
        function _collapseK(uq)
            pcs = map(uq.profiles) do pc
                pc.parameter_name == "K" || return pc
                ProfileCurve(pc.parameter_index, pc.parameter_name, pc.profile_values,
                    pc.log_likelihoods, pc.optimal_parameters, pc.ci_lower, pc.ci_lower,
                    pc.threshold, pc.reference_ll)
            end
            ProfileLikelihoodResult{Float64}(pcs, uq.fit_result, uq.times, uq.method)
        end
        uq_zero    = copy(uq_results)
        uq_zero[2] = _collapseK(uq_results[2])
        post = buildPosterior(sm, data, uq_zero, cm_params; box_score = :jaccard)
        @test post.raw_scores[2] == 0.0
    end

    @testset "buildPosterior gate and min_score = :elbow" begin
        # K's upper CI side opened on cm_param_sets 2 and 3: they fail the gate (the data's CI is
        # bounded everywhere), whatever their score
        # (cm_param_set 3 is the generating one and scores highest).
        _openK(uq) = let
            pcs = map(uq.profiles) do pc
                pc.parameter_name == "K" || return pc
                ProfileCurve(pc.parameter_index, pc.parameter_name, pc.profile_values,
                    pc.log_likelihoods, pc.optimal_parameters, pc.ci_lower, nothing,
                    pc.threshold, pc.reference_ll)
            end
            ProfileLikelihoodResult{Float64}(pcs, uq.fit_result, uq.times, uq.method)
        end
        uq_open = copy(uq_results)
        uq_open[2] = _openK(uq_results[2])
        uq_open[3] = _openK(uq_results[3])
        @test openSides(uq_open[2]) == [false false; false true]     # rows r, K; columns lower, upper
        @test openSides(uq_open[4]) == falses(2, 2)

        match = IdentifiabilityMatchGate()
        @testset "bridge = $bridge" for bridge in (:box_overlap, :data_trace_in_box, :symmetric_trace)
            raw   = buildPosterior(sm, data, uq_open, cm_params; bridge = bridge)
            gated = buildPosterior(sm, data, uq_open, cm_params; bridge = bridge, gate = match)
            @test raw.gate === NullGate()
            @test gated.gate === match
            @test raw isa CMPosteriorResult{LinearCIInterp,NullGate}
            @test gated isa CMPosteriorResult{LinearCIInterp,IdentifiabilityMatchGate}
            @test openSides(raw.data_profiles) == falses(2, 2)
            @test raw.raw_scores[3] > 0                   # the gate, not the bridge, removes it

            # The posterior keeps the bridge's scores and records who passes.
            @test gated.raw_scores == raw.raw_scores
            @test gated.gate_pass == [true, false, false, true, true]
            @test all(raw.gate_pass)
            expected = [raw.raw_scores[1], 0.0, 0.0, raw.raw_scores[4], raw.raw_scores[5]]
            @test gated.raw_scores[gated.gate_pass] == raw.raw_scores[[1, 4, 5]]

            # Everything aligned with the CM grid treats a failing cm_param_set as score 0.
            @test scoreGrid(gated) == expected
            @test gated.accepted == (expected .> 0)
            @test !any(gated.accepted[.!gated.gate_pass])
            @test posteriorSamples(gated) == cm_params[gated.accepted, :]
            w = posteriorWeights(gated)
            @test w[2] == 0.0 && w[3] == 0.0
            @test sum(expected) == 0 ? all(iszero, w) : w ≈ expected ./ sum(expected)
            @test scoreGrid(raw) == raw.raw_scores

            # applyGate(post): the stored gate's result, in full.
            g = applyGate(gated)
            @test g.scores == expected
            @test g.pass == gated.gate_pass
            @test g.mismatch[2, :, :] == [false false; false true]
            @test size(g.mismatch) == (5, 2, 2)
            n = applyGate(raw)
            @test n.scores == raw.raw_scores
            @test all(n.pass)
            @test !haskey(n, :mismatch)

            # applyGate(gate, post): another gate on the same raw scores; post is not changed.
            @test applyGate(match, raw) == g
            @test raw.gate === NullGate() && all(raw.gate_pass)
            @test applyGate(NullGate(), gated).scores == gated.raw_scores
            @test gated.gate_pass == [true, false, false, true, true]

            # min_score = :elbow: τ_elbow of the scores that pass the gate, stored on the result.
            el = buildPosterior(sm, data, uq_open, cm_params; bridge = bridge, gate = match, min_score = :elbow)
            @test el.raw_scores == raw.raw_scores
            @test el.gate_pass == gated.gate_pass
            @test el.min_score == elbowThreshold(el.raw_scores[el.gate_pass]).tau
            @test el.min_score == elbowThreshold(expected).tau
            @test el.accepted == el.gate_pass .& (el.raw_scores .> el.min_score)
            el_raw = buildPosterior(sm, data, uq_open, cm_params; bridge = bridge, min_score = :elbow)
            @test el_raw.min_score == elbowThreshold(raw.raw_scores).tau
            @test el_raw.raw_scores == raw.raw_scores
        end

        @testset "interior queries apply the gate: $bridge" for bridge in (:box_overlap, :data_trace_in_box)
            raw   = buildPosterior(sm, data, uq_open, cm_params; bridge = bridge)
            gated = buildPosterior(sm, data, uq_open, cm_params; bridge = bridge, gate = match)
            # At the sampled CM points an interior query reproduces the stored (gated) score.
            for (k, c) in enumerate(cvals)
                @test posteriorScore(gated, [c]) == scoreGrid(gated)[k]
            end
            # Between two CM param_sets that are open in K above (2 and 3): the interpolated
            # upper bound is at the edge of the swept range → open → fails (data is bounded).
            @test posteriorScore(raw, [2.999]) > 0
            @test posteriorScore(gated, [2.999]) == 0.0
            @test !inPosterior(gated, [2.999])
            # Between an open (3) and a bounded (4) CM param_set: the interpolated bound is
            # strictly inside the swept range → bounded → passes, score as without the gate.
            bridge === :box_overlap && @test posteriorScore(raw, [3.001]) > 0
            @test posteriorScore(gated, [3.001]) == posteriorScore(raw, [3.001])
            # Between two bounded CM param_sets (4 and 5): unchanged.
            @test posteriorScore(gated, [4.5]) == posteriorScore(raw, [4.5])
            # Batch form
            @test posteriorScore(gated, reshape([2.999, 3.001, 4.5], :, 1)) ==
                  [0.0, posteriorScore(raw, [3.001]), posteriorScore(raw, [4.5])]
        end

        @testset "profile range: the prior range, or ProfileLikelihood's bounds" begin
            # Without `bounds` the profile range is the SM prior range.
            @test SmoreFit._profileRange(uq_results[1]) == ([0.01, 0.5], [2.0, 10.0])

            # Profiles run with `bounds` narrower than the prior in K (upper edge 6 instead of 10).
            narrow = ParameterPrior([0.01, 0.5], [2.0, 6.0]; names = ["r", "K"])
            opts   = ProfileLikelihood(n_points = 12, bounds = narrow)
            uq_b   = map(cvals) do c
                d    = CMData(μ = vec(_logistic(t, [rtrue(c), Ktrue], nothing)), σ = noise .* ones(length(t)), times = t)
                prob = SMFitProblem(sm, d, sm_prior)
                quantifyUncertainty(opts, prob, fitSurrogate(prob, P0), 1)
            end
            @test SmoreFit._profileRange(uq_b[1]) == ([0.01, 0.5], [2.0, 6.0])
            uq_b[2] = _openK(uq_b[2]); uq_b[3] = _openK(uq_b[3])

            raw   = buildPosterior(sm, data, uq_b, cm_params)
            gated = buildPosterior(sm, data, uq_b, cm_params; gate = match)
            # An open side is stored at the edge of `bounds`, not of the prior.
            @test gated.ub_table[3, 2] == 6.0
            @test gated.gate_pass == [true, false, false, true, true]
            # Interior gating compares against that edge: between the two open CM param_sets the
            # interpolated K upper bound is 6.0 → open → fails; next to a bounded one it passes.
            @test posteriorScore(raw, [2.999]) > 0
            @test posteriorScore(gated, [2.999]) == 0.0
            @test posteriorScore(gated, [3.001]) == posteriorScore(raw, [3.001])
            for (k, c) in enumerate(cvals)
                @test posteriorScore(gated, [c]) == scoreGrid(gated)[k]
            end
        end

        # A gate with no interior method: grid scores work, interior queries say why they cannot.
        post_custom = buildPosterior(sm, data, uq_results, cm_params; gate = _KeepFirstGate())
        @test post_custom.gate_pass == [true, false, false, false, false]
        @test scoreGrid(post_custom)[2:end] == zeros(4)
        @test applyGate(post_custom).pass == post_custom.gate_pass
        @test post_custom isa CMPosteriorResult{LinearCIInterp,_KeepFirstGate}
        @test_throws ArgumentError posteriorScore(post_custom, [2.5])

        # defaults are unchanged
        post = buildPosterior(sm, data, uq_results, cm_params)
        @test post.gate === NullGate()
        @test post.min_score == 0.0
        @test_throws ArgumentError buildPosterior(sm, data, uq_results, cm_params; min_score = :knee)
    end

    @testset "validation" begin
        # data must have a single param set
        bad = CMData(μ = rand(length(t), 2), σ = ones(length(t), 2), times = t, cm_param_sets = 2)
        @test_throws ArgumentError buildPosterior(sm, bad, uq_results, cm_params)

        # uq_results length must match the number of cm_param_sets
        @test_throws ArgumentError buildPosterior(sm, data, uq_results[1:3], cm_params)

        # unknown bridge method
        @test_throws ArgumentError buildPosterior(sm, data, uq_results, cm_params; bridge = :nope)

        # unknown box_score, with any bridge
        @test_throws ArgumentError buildPosterior(sm, data, uq_results, cm_params; box_score = :union)
        @test_throws ArgumentError buildPosterior(sm, data, uq_results, cm_params; bridge = :data_trace_in_box, box_score = :union)
    end
end

@testset "identifiability-match rule on CI shapes" begin
    scores    = [0.9, 0.5, 0.3, 0.0, 0.7]
    open_data = [false false; false true]                # rows r, K; columns lower, upper: K open above
    flags = Bool[                                        # r_lo r_hi K_lo K_hi, one row per CM param_set
        0 0 0 1      # same shape → pass
        0 0 0 0      # bounded where the data's is open → fail
        1 0 0 1      # open where the data's is bounded → fail
        0 0 0 1      # pass (score already 0)
        0 1 0 0      # two sides differ → fail
    ]
    open_grid = _sideArray(flags)                        # [5 × 2 × 2]
    g = SmoreFit._matchShapes(scores, open_grid, open_data)
    @test g.scores == [0.9, 0.0, 0.0, 0.0, 0.0]
    @test g.scores isa Vector{Float64}
    @test g.pass == [true, false, false, true, false]
    @test g.mismatch == _sideArray(Bool[0 0 0 0; 0 0 0 1; 1 0 0 0; 0 0 0 0; 0 1 0 1])
    @test g.mismatch[5, 1, 2] && g.mismatch[5, 2, 2]     # CM param_set 5: r upper and K upper differ
    @test scores == [0.9, 0.5, 0.3, 0.0, 0.7]            # input untouched

    # integer scores and BitArray flags
    gi = SmoreFit._matchShapes([1, 2], _sideArray(BitMatrix([0 0; 0 1])), falses(1, 2))
    @test gi.scores == [1.0, 0.0]

    @test_throws DimensionMismatch SmoreFit._matchShapes(scores, open_grid[1:4, :, :], open_data)
    @test_throws DimensionMismatch SmoreFit._matchShapes(scores, open_grid, open_data[1:1, :])

    @test openSides([0.1, 5.0], [0.3, nothing]) == [false false; false true]
    @test openSides([0.1, 5.0], [0.3, nothing]) isa BitMatrix
    @test openSides([missing, 5.0], [0.3, 9.0]) == [true false; false false]
    @test_throws DimensionMismatch openSides([0.1], [0.3, 1.0])
end

@testset "elbowThreshold" begin
    # A sharp elbow: the point farthest below the line from (rank 1, 1.0) to (rank 6, 0.01).
    e = elbowThreshold([0.02, 1.0, 0.0, 0.9, 0.05, 0.01, 0.0, 0.8])
    @test e.tau == 0.05
    @test e.rank == 4
    @test e.n_positive == 6
    @test e.note == ""

    # Order and non-positive scores do not matter.
    @test elbowThreshold([0.8, -1.0, 0.01, 0.05, 0.9, 0.02, 1.0]).tau == 0.05

    # Tied scores share one fate: τ is a score value and acceptance is strict.
    s = [1.0, 0.9, 0.1, 0.1, 0.1, 0.05]
    et = elbowThreshold(s)
    @test et.tau == 0.1
    @test count(s .> et.tau) == 2

    # τ_elbow = 0: fewer than three positive scores, all equal, or nothing below the line.
    for s0 in (Float64[], [0.0, 0.0], [0.4], [0.4, 0.2, 0.0])
        e0 = elbowThreshold(s0)
        @test e0.tau == 0.0 && e0.rank == 0
        @test e0.note == "fewer than three positive scores"
        @test e0.n_positive == count(>(0), s0)
    end
    @test elbowThreshold([0.3, 0.3, 0.3]).note == "no point below the line (all positive scores equal)"
    @test elbowThreshold([0.3, 0.3, 0.3]).tau == 0.0
    concave = elbowThreshold([1.0, 0.95, 0.9, 0.1])      # every interior point above the line
    @test concave.tau == 0.0
    @test concave.note == "no point below the line"
    @test elbowThreshold([3, 2, 1]).tau === 0.0          # on the line; integer input
end

# Gate, then elbow, on a 3 × 4 CM grid (12 CM param_sets), with every number worked by hand.
# The data's CI is bounded in r and open above in K: [false, false, false, true].
@testset "gate + elbow on a 3 × 4 grid" begin
    open_data = [false false; false true]
    #             score  r_lo   r_hi   K_lo   K_hi
    grid = [
        (0.90, false, false, false, true),    #  1  pass
        (0.80, false, false, false, true),    #  2  pass
        (0.85, false, false, false, false),   #  3  K bounded above → fail
        (0.70, false, false, false, true),    #  4  pass
        (0.10, false, false, false, true),    #  5  pass  ← elbow
        (0.60, true,  false, false, true),    #  6  r open below → fail
        (0.08, false, false, false, true),    #  7  pass
        (0.05, false, false, false, true),    #  8  pass
        (0.00, false, false, false, true),    #  9  pass, score 0
        (0.04, false, false, false, true),    # 10  pass
        (0.30, false, true,  false, false),   # 11  two sides differ → fail
        (0.02, false, false, false, true),    # 12  pass
    ]
    scores    = [g[1] for g in grid]
    open_grid = _sideArray([g[c] for g in grid, c in 2:5])     # [12 × 2 × 2]
    reference = BitVector([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])      # a "true" set, for the Jaccard index
    jaccard(a, b) = count(a .& b) / count(a .| b)

    g = SmoreFit._matchShapes(scores, open_grid, open_data)
    @test findall(.!g.pass) == [3, 6, 11]
    @test g.mismatch[11, :, :] == [false true; false true]      # r upper and K upper
    @test g.scores == [0.90, 0.80, 0.0, 0.70, 0.10, 0.0, 0.08, 0.05, 0.0, 0.04, 0.0, 0.02]

    # Stage 1, no filter: 11 positive raw scores. Stage 2, gate only: 8.
    @test count(>(0), scores) == 11
    @test count(>(0), g.scores) == 8
    @test jaccard(scores .> 0, reference) ≈ 3 / 11
    @test jaccard(g.scores .> 0, reference) ≈ 3 / 8

    # Stage 3, gate + elbow. Sorted passing scores: 0.90 0.80 0.70 0.10 0.08 0.05 0.04 0.02;
    # the point farthest below the line from (1, 0.90) to (8, 0.02) is rank 4, score 0.10.
    e = elbowThreshold(scores[g.pass])
    @test e == elbowThreshold(g.scores)             # a failing CM param_set counts as score 0
    @test e.tau == 0.10
    @test e.rank == 4
    @test e.n_positive == 8
    accepted = g.pass .& (scores .> e.tau)
    @test findall(accepted) == [1, 2, 4]
    @test jaccard(accepted, reference) == 1.0

    # Without the gate the elbow lands elsewhere and keeps two CM param_sets the gate removes.
    e_raw = elbowThreshold(scores)
    @test e_raw.tau == 0.10
    @test findall(scores .> e_raw.tau) == [1, 2, 3, 4, 6, 11]
end
