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
    Sigma0 = diagm(ones(sum(labeling)))
    sample_std = 1.0
    predictive = [2]
    prognostic = [[3]]
    policy = BiasedCoinPolicyLinear(n, m, theta0, Sigma0, sample_std, predictive, prognostic, labeling)
    ContextualBandits.initialize!(policy)
    @test policy.model.theta_t == theta0
    @test policy.model.Sigma_t == Sigma0
    rng = Xoshiro(1234)
    Xcurrent = rand(rng, FX)
    X = rand(rng, FX, 200)
    W = rand(rng, 1:n, 200)
    Y = randn(rng, 200)
    for i in eachindex(Y)
        ContextualBandits.state_update!(policy, W[1:i], view(X,:,1:i), Y[1:i], rng)
    end
    w = ContextualBandits.allocation(policy, Xcurrent, W, X, Y, rng)
    @test w in 1:n
    Ycurrent = randn(rng)
    ContextualBandits.state_update!(policy, w, Xcurrent, Ycurrent, rng)
    @test policy.model.theta_t ≈ BayesUpdateNormal(theta0, Sigma0, interact([W;w], n, hcat(X,Xcurrent), labeling), [Y; Ycurrent], sample_std)[1]
    @test all(ContextualBandits.implementation(policy, Xcurrent, W, X, Y) .∈ Ref(1:3))

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
    Sigma0 = diagm(ones(sum(labeling)))
    sample_std = 1.0
    predictive = [2]
    prognostic = [[3]]
    policy = RABC_OCBA_PolicyLinear(n, m, theta0, Sigma0, sample_std, predictive, prognostic, labeling)
    ContextualBandits.initialize!(policy)
    @test policy.model.theta_t == theta0
    @test policy.model.Sigma_t == Sigma0
    rng = Xoshiro(1234)
    Xcurrent = rand(rng, FX)
    X = rand(rng, FX, 200)
    W = rand(rng, 1:n, 200)
    Y = randn(rng, 200)
    for i in eachindex(Y)
        ContextualBandits.state_update!(policy, W[1:i], view(X,:,1:i), Y[1:i], rng)
    end
    w = ContextualBandits.allocation(policy, Xcurrent, W, X, Y, rng)
    @test w in 1:n
    Ycurrent = randn(rng)
    ContextualBandits.state_update!(policy, w, Xcurrent, Ycurrent, rng)
    @test policy.model.theta_t ≈ BayesUpdateNormal(theta0, Sigma0, interact([W;w], n, hcat(X,Xcurrent), labeling), [Y; Ycurrent], sample_std)[1]
    @test all(ContextualBandits.implementation(policy, Xcurrent, W, X, Y) .∈ Ref([0,1]))

    policy = RABC_OCBA_PolicyLinear(n, m, theta0, Sigma0, sample_std, FX, labeling)
    ContextualBandits.initialize!(policy)
    @test policy.model.theta_t == theta0
    @test policy.model.Sigma_t == Sigma0
    @test policy.predictive == [2]
    @test policy.prognostic == [[3]]
end
