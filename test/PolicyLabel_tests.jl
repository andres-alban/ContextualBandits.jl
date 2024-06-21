using ContextualBandits
using Random
using LinearAlgebra
using Test

@testset "RandomPolicyLabel" begin
    Wn = 2
    m = 3
    sample_std = 1.0
    labels = [true, false, false, true, true, false, true, false, true]
    theta0 = zeros(sum(labels))
    Sigma0 = Diagonal(ones(sum(labels)))
    policy = RandomPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)
    ContextualBandits.initialize!(policy,[],[],[])
    @test policy.theta_t == [0.0, 0.0, 0.0, 0.0, 0.0]
    @test policy.Sigma_t == Diagonal(ones(sum(labels)))
    X = [1.0, 2.0, 5.0]
    W = [1]
    Y = [1.0]
    ContextualBandits.state_update!(policy,W,X,Y)
    @test policy.theta_t == [0.14285714285714285, 0.14285714285714285, 0.2857142857142857, 0.0, 0.0]
    rng = MersenneTwister(1234)
    @test ContextualBandits.allocation(policy,X,W,X,Y,rng) == 1
    @test ContextualBandits.implementation(policy,X,W,X,Y) == [1]
    @test ContextualBandits.policy_labeling(policy) == labels
end

@testset "GreedyPolicyLabel" begin
    Wn = 2
    m = 3
    sample_std = 1.0
    labels = [true, false, false, true, true, false, true, false, true]
    theta0 = zeros(sum(labels))
    Sigma0 = Diagonal(ones(sum(labels)))
    policy = GreedyPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)
    ContextualBandits.initialize!(policy,[],[],[])
    @test policy.theta_t == [0.0, 0.0, 0.0, 0.0, 0.0]
    @test policy.Sigma_t == Diagonal(ones(sum(labels)))
    X = [1.0, 2.0, 5.0]
    W = [1]
    Y = [1.0]
    ContextualBandits.state_update!(policy,W,X,Y)
    @test policy.theta_t == [0.14285714285714285, 0.14285714285714285, 0.2857142857142857, 0.0, 0.0]
    rng = MersenneTwister(1234)
    @test ContextualBandits.allocation(policy,X,W,X,Y,rng) == 1
    @test ContextualBandits.implementation(policy,X,W,X,Y) == [1]
    @test ContextualBandits.policy_labeling(policy) == labels
end