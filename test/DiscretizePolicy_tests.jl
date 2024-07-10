using ContextualBandits
using Distributions
using LinearAlgebra
using Random
using Test

@testset "DiscretizePolicy" begin
    n = 3
    FX = CovariatesIndependent([Normal(), Normal()])
    m = length(FX)
    theta0 = zeros(n*m)
    Sigma0 = Diagonal(ones(n*m))
    sample_std = 1.0
    breakpoints = [[-1,1], [-2,-1,0,1,2]]
    FX_discretized, v, gn = discretizeFX(FX, breakpoints)
    fEVI = fEVIDiscrete(n, m, theta0, Sigma0, sample_std, FX_discretized)
    policy = DiscretizePolicy(fEVI, FX, breakpoints)
    random_policy = RandomPolicyLinear(n, m, theta0, Sigma0, sample_std)

    T = 10
    policies = Dict("fEVI_discretized" => policy, "random" => random_policy)
    rng = MersenneTwister(1234)
    outcome_model = OutcomeLinear(n, m, rand(rng,n*m), sample_std)
    results = simulation_stochastic(10, FX, n, T, policies, outcome_model, post_reps=10, rng=rng)
    eocoff = results["output"]["fEVI_discretized"]["regret_off"]["mean"]
    eocon = results["output"]["fEVI_discretized"]["regret_on"]["mean"]
    @test !all(eocoff .== eocoff[1])
    @test !all(eocon .== eocon[1])
end