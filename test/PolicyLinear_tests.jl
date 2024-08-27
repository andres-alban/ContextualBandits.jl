using ContextualBandits
using Random
using LinearAlgebra
using Test

@testset "RandomPolicyLinear" begin
    n = 2
    m = 3
    sample_std = 1.0
    labeling = [true, false, false, true, true, false, true, false, true]
    theta0 = zeros(sum(labeling))
    Sigma0 = diagm(ones(sum(labeling)))
    policy = RandomPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    ContextualBandits.initialize!(policy, [], [], [])
    @test policy.model.theta_t == [0.0, 0.0, 0.0, 0.0, 0.0]
    @test policy.model.Sigma_t == Diagonal(ones(sum(labeling)))
    X = [1.0, 2.0, 5.0]
    W = 1
    Y = 1.0
    ContextualBandits.state_update!(policy, W, X, Y)
    @test policy.model.theta_t == BayesUpdateNormal(theta0, Sigma0, interact(W, n, X, labeling), Y, sample_std)[1]
    rng = Xoshiro(1234)
    w = ContextualBandits.allocation(policy, X, W, X, Y, rng)
    @test w in 1:n
    wpost = ContextualBandits.implementation(policy, X, W, X, Y)
    @test all(wpost .∈ Ref(1:n))
    @test ContextualBandits.policy_labeling(policy) == labeling
    rng = Xoshiro(1234)
    @test ContextualBandits.allocationIndependent(policy, X, [W], X, [Y], rng) == w
    @test ContextualBandits.implementationIndependent(policy, X, [W], X, [Y]) == wpost

    ContextualBandits.initialize!(policy, W, X, Y)
    theta_exp, Sigma_exp = BayesUpdateNormal(theta0, Sigma0, interact(W, n, X, labeling), Y, sample_std)
    robustify_prior_linear!(theta_exp, Sigma_exp, n, m, labeling)
    @test policy.model.theta_t == theta_exp
    @test policy.model.Sigma_t == Sigma_exp
end

@testset "GreedyPolicyLinear" begin
    n = 2
    m = 3
    sample_std = 1.0
    labeling = [true, false, false, true, true, false, true, false, true]
    theta0 = zeros(sum(labeling))
    Sigma0 = diagm(ones(sum(labeling)))
    policy = GreedyPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    ContextualBandits.initialize!(policy, [], [], [])
    @test policy.model.theta_t == [0.0, 0.0, 0.0, 0.0, 0.0]
    @test policy.model.Sigma_t == Diagonal(ones(sum(labeling)))
    X = [1.0, 2.0, 5.0]
    W = [1]
    Y = [1.0]
    ContextualBandits.state_update!(policy, W, X, Y)
    @test policy.model.theta_t == BayesUpdateNormal(theta0, Sigma0, interact(W, n, X, labeling), Y, sample_std)[1]
    rng = Xoshiro(1234)
    @test ContextualBandits.allocation(policy, X, W, X, Y, rng) in 1:n
    @test all(ContextualBandits.implementation(policy, X, W, X, Y) .∈ Ref(1:3))
    @test ContextualBandits.policy_labeling(policy) == labeling
end