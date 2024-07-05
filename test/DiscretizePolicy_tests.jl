using ContextualBandits
using Distributions
using LinearAlgebra
using Random
using Test

@testset "DiscretizePolicy" begin
    Wn = 3
    FX = CovariatesIndependent([Normal(), Normal()])
    m = length(FX)
    theta0 = zeros(Wn*m)
    Sigma0 = Diagonal(ones(Wn*m))
    sample_std = 1.0
    breakpoints = [[-1,1], [-2,-1,0,1,2]]
    FX_discretized, v, gn = discretizeFX(FX, breakpoints)
    fEVI = fEVIDiscrete(Wn, m, theta0, Sigma0, sample_std, FX_discretized)
    policy = DiscretizePolicy(fEVI, FX, breakpoints)
    random_policy = RandomPolicyLinear(Wn, m, theta0, Sigma0, sample_std)

    T = 10
    delay = 0
    policies = Dict("fEVI_discretized" => policy, "random" => random_policy)
    rng = MersenneTwister(1234)
    outcome_model = OutcomeLinearFixed(Wn, m, rand(rng,Wn*m), sample_std)
    results = simulation_stochastic(FX, FX, Wn, T, delay, policies, outcome_model, reps=10, post_reps=10, rng=rng)
    eocoff = results["output"]["fEVI_discretized"]["EOC_off"]["mean"]
    eocon = results["output"]["fEVI_discretized"]["EOC_on"]["mean"]
    @test !all(eocoff .== eocoff[1])
    @test !all(eocon .== eocon[1])
end