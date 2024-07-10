using ContextualBandits
using Random
using Distributions
using LinearAlgebra
using Test

@testset "BiasedCoin policy" begin
    n = 3
    FX = CovariatesIndependent([Categorical(1/2,1/2), Categorical(1/2,1/2)])
    m = length(FX)
    labeling=[false, false, true, true, true, false, true, true, false, true, true, false]
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    predictive = [2]
    prognostic = [[3]]
    policy = BiasedCoinPolicyLinear(n, m, theta0, Sigma0, sample_std, predictive, prognostic, labeling)
    ContextualBandits.initialize!(policy)
    @test policy.model.theta_t == theta0
    @test policy.model.Sigma_t == Sigma0
    rng = MersenneTwister(1234)
    Xcurrent = rand(rng, FX)
    X = rand(rng, FX, 200)
    W = rand(rng, 1:n, 200)
    Y = randn(rng, 200)
    for i in eachindex(Y)
        ContextualBandits.state_update!(policy, W[1:i], view(X,:,1:i), Y[1:i], rng)
    end
    w = ContextualBandits.allocation(policy, Xcurrent, W, X, Y, rng)
    @test w == 2
    ContextualBandits.state_update!(policy, w, Xcurrent, randn(rng), rng)
    @test policy.model.theta_t == [0.32043894256754457, -0.2868626983183138, 0.2216002700633016, -0.3764185200729958, 0.32565621625021324, 0.09583575552810948, -0.2817055714305785]
    @test ContextualBandits.implementation(policy, Xcurrent, W, X, Y) == [2]

    policy = BiasedCoinPolicyLinear(n, m, theta0, Sigma0, sample_std, FX, labeling)
    ContextualBandits.initialize!(policy)
    @test policy.model.theta_t == theta0
    @test policy.model.Sigma_t == Sigma0
    @test policy.predictive == [2]
    @test policy.prognostic == [[3]]
end

@testset "RABC policy" begin
    n = 3
    FX = CovariatesIndependent([Categorical(1/2,1/2), Categorical(1/2,1/2)])
    m = length(FX)
    labeling=[false, false, true, true, true, false, true, true, false, true, true, false]
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    predictive = [2]
    prognostic = [[3]]
    policy = RABC_OCBA_PolicyLinear(n, m, theta0, Sigma0, sample_std, predictive, prognostic, labeling)
    ContextualBandits.initialize!(policy)
    @test policy.model.theta_t == theta0
    @test policy.model.Sigma_t == Sigma0
    rng = MersenneTwister(1234)
    Xcurrent = rand(rng, FX)
    X = rand(rng, FX, 200)
    W = rand(rng, 1:n, 200)
    Y = randn(rng, 200)
    for i in eachindex(Y)
        ContextualBandits.state_update!(policy, W[1:i], view(X,:,1:i), Y[1:i], rng)
    end
    w = ContextualBandits.allocation(policy, Xcurrent, W, X, Y, rng)
    @test w == 2
    ContextualBandits.state_update!(policy, w, Xcurrent, randn(rng), rng)
    @test policy.model.theta_t == [0.32043894256754457, -0.2868626983183138, 0.2216002700633016, -0.3764185200729958, 0.32565621625021324, 0.09583575552810948, -0.2817055714305785]
    @test ContextualBandits.implementation(policy, Xcurrent, W, X, Y) == [2]

    policy = RABC_OCBA_PolicyLinear(n, m, theta0, Sigma0, sample_std, FX, labeling)
    ContextualBandits.initialize!(policy)
    @test policy.model.theta_t == theta0
    @test policy.model.Sigma_t == Sigma0
    @test policy.predictive == [2]
    @test policy.prognostic == [[3]]
end
