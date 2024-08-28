using ContextualBandits
using Distributions
using LinearAlgebra
using Random
using Test

@testset "RandEpsilon with fEVIDiscrete" begin
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

    epsilon = 0.25
    fEVI_rand = RandEpsilon(fEVI, n, epsilon)
    policies = Dict("fEVI_rand" => fEVI_rand, "fEVI" => fEVI)

    rng = Xoshiro(1234)
    outcome_model = OutcomeLinear(n, m, rand(rng, sum(labeling)), sample_std, labeling)

    T = 100
    results = simulation_stochastic(10, FX, n, T, policies, outcome_model, post_reps=10, rng=rng)

    # Test that the labeling is not the same for all time points or trivial at the end
    x = results["output"]["fEVI_rand"]
    @test !all(x["regret_off"]["mean"] .== x["regret_off"]["mean"][1])
    @test !all(x["regret_on"]["mean"] .== x["regret_on"]["mean"][1])
end

@testset "RandEpsilon with fEVI_MC_PolicyLinear" begin
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

    epsilon = 0.25
    fEVIMC_rand = RandEpsilon(fEVIMC, n, epsilon)
    policies = Dict("fEVIMC_rand" => fEVIMC_rand, "fEVIMC" => fEVIMC)

    rng = Xoshiro(1234)
    outcome_model = OutcomeLinear(n, m, rand(rng, sum(labeling)), sample_std, labeling)

    T = 100
    results = simulation_stochastic(10, FX, n, T, policies, outcome_model, post_reps=10, rng=rng)

    # Test that the labeling is not the same for all time points or trivial at the end
    x = results["output"]["fEVIMC_rand"]
    @test !all(x["regret_off"]["mean"] .== x["regret_off"]["mean"][1])
    @test !all(x["regret_on"]["mean"] .== x["regret_on"]["mean"][1])
end
