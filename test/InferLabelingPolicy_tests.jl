using ContextualBandits
using Distributions
using LinearAlgebra
using Random
using Test

@testset "InferLabelingPolicy with fEVIDiscrete" begin
    Wn = 3
    FX = CovariatesIndependent([Categorical(1/2,1/2), Categorical(1/2,1/2)])
    m = length(FX)
    labeling=[false, false, true, true, true, false, true, true, false, true, true, false]
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    etaon = 1
    etaoff = 10
    fEVI = fEVIDiscrete(Wn, m, theta0, Sigma0, sample_std, FX, labeling)

    labeling_selector = LassoCVLabelingSelector(Wn,m)
    T = 100
    schedule = 1:10:T

    fEVI_infer = InferLabelingPolicy(fEVI,labeling_selector,schedule)
    policies = Dict( "fEVI_infer" => fEVI_infer, "fEVI" => fEVI)

    rng = MersenneTwister(1234)
    outcome_model = OutcomeLinearFixed(Wn, m, rand(rng,sum(labeling)), sample_std, labeling)

    results = simulation_stochastic(FX,FX,Wn,T,0,policies,outcome_model,reps=10,post_reps=10,rng=rng)

    # Test that the labeling is not the same for all time points or trivial at the end
    x = results["output"]["fEVI_infer"]
    @test !all(x["sum_labeling"]["mean"] .== x["sum_labeling"]["mean"][1])
    @test !all(x["labeling_frac"]["mean"] .== x["labeling_frac"]["mean"][1])
end

@testset "InferLabelingPolicy with fEVI_MC_PolicyLinear" begin
    Wn = 3
    FX = CovariatesIndependent([Normal() for _ in 1:20])
    m = length(FX)
    labeling = rand(Bool, (Wn+1)*m)
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    etaon = 1
    etaoff = 10
    fEVIMC = fEVI_MC_PolicyLinear(Wn, m, theta0, Sigma0, sample_std, FX, etaon, etaoff, labeling)

    labeling_selector = LassoCVLabelingSelector(Wn,m)
    T = 100
    schedule = 1:10:T

    fEVIMC_infer = InferLabelingPolicy(fEVIMC,labeling_selector,schedule)
    policies = Dict("fEVIMC_infer" => fEVIMC_infer, "fEVIMC" => fEVIMC)

    rng = MersenneTwister(1234)
    outcome_model = OutcomeLinearFixed(Wn, m, rand(rng,sum(labeling)), sample_std, labeling)

    results = simulation_stochastic(FX,FX,Wn,T,0,policies,outcome_model,reps=10,post_reps=10,rng=rng)

    # Test that the labeling is not the same for all time points or trivial at the end
    x = results["output"]["fEVIMC_infer"]
    @test !all(x["sum_labeling"]["mean"] .== x["sum_labeling"]["mean"][1])
    @test !all(x["labeling_frac"]["mean"] .== x["labeling_frac"]["mean"][1])
end
