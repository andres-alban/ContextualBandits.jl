using ContextualBandits
using Distributions
using LinearAlgebra
using Random
using Test

@testset "InferLabelingPolicy with fEVIDiscrete" begin
    n = 3
    FX = CovariatesIndependent([Categorical(1 / 2, 1 / 2), Categorical(1 / 2, 1 / 2)])
    m = length(FX)
    labeling = [false, false, true, true, true, false, true, true, false, true, true, false]
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    etaon = 1
    etaoff = 10
    fEVI = fEVIDiscrete(n, m, theta0, Sigma0, sample_std, FX, labeling)

    labeling_selector = LassoCVLabelingSelector(n, m)
    T = 100
    schedule = 1:10:T

    fEVI_infer = InferLabelingPolicy(fEVI, labeling_selector, schedule)
    policies = Dict("fEVI_infer" => fEVI_infer, "fEVI" => fEVI)

    rng = Xoshiro(1234)
    outcome_model = OutcomeLinear(n, m, rand(rng, sum(labeling)), sample_std, labeling)

    results = simulation_stochastic(10, FX, n, T, policies, outcome_model, post_reps=10, rng=rng)

    # Test that the labeling is not the same for all time points or trivial at the end
    x = results["output"]["fEVI_infer"]
    @test !all(x["sum_labeling"]["mean"] .== x["sum_labeling"]["mean"][1])
    @test !all(x["labeling_frac"]["mean"] .== x["labeling_frac"]["mean"][1])
end

@testset "InferLabelingPolicy with fEVI_MC_PolicyLinear" begin
    n = 3
    FX = CovariatesIndependent([Normal() for _ in 1:20])
    m = length(FX)
    labeling = rand(Bool, (n + 1) * m)
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    etaon = 1
    etaoff = 10
    fEVIMC = fEVI_MC_PolicyLinear(n, m, theta0, Sigma0, sample_std, FX, etaon, etaoff, labeling)

    labeling_selector = LassoCVLabelingSelector(n, m)
    T = 100
    schedule = 1:10:T

    fEVIMC_infer = InferLabelingPolicy(fEVIMC, labeling_selector, schedule)
    policies = Dict("fEVIMC_infer" => fEVIMC_infer, "fEVIMC" => fEVIMC)

    rng = Xoshiro(1234)
    outcome_model = OutcomeLinear(n, m, rand(rng, sum(labeling)), sample_std, labeling)

    results = simulation_stochastic(10, FX, n, T, policies, outcome_model, post_reps=10, rng=rng)

    # Test that the labeling is not the same for all time points or trivial at the end
    x = results["output"]["fEVIMC_infer"]
    @test !all(x["sum_labeling"]["mean"] .== x["sum_labeling"]["mean"][1])
    @test !all(x["labeling_frac"]["mean"] .== x["labeling_frac"]["mean"][1])
end
