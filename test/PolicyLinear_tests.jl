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
    @test allocation_current(policy, X, [W], X, [Y], rng) == w
    @test implementation_current(policy, X, [W], X, [Y]) == wpost

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

@testset "allocation_current" begin
    n = 5
    m = 2
    sample_std = 1.0
    labeling = vcat([true], falses(m * (n + 1) - 1))
    theta0 = zeros(sum(labeling))
    Sigma0 = diagm(ones(sum(labeling)))
    policy = RandomPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    ContextualBandits.initialize!(policy)


    T = 4
    X = randn(m, T)
    Y = randn(T)
    W1 = zeros(Int, T)
    for t in 1:T
        rng1 = Xoshiro(2024)
        W1[t] = @test_nowarn allocation_current(policy, X[:, t], W1[1:t-1], X[:, 1:t-1], Y[1:t-1], rng1, true)
    end

    W2 = zeros(Int, T)
    for t in 1:T
        rng2 = Xoshiro(2024)
        W2[t] = allocation_current(policy, X[:, t], W2[1:t-1], X[:, 1:t-1], Y[1:t-1], rng2, false)
    end

    @test W1 == W2
end