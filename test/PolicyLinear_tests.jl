using ContextualBandits
using Random
using LinearAlgebra
using Test

@testset "RandomPolicyLinear" begin
    Wn = 2
    m = 3
    sample_std = 1.0
    labeling = [true, false, false, true, true, false, true, false, true]
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    policy = RandomPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling)
    ContextualBandits.initialize!(policy,[],[],[])
    @test policy.model.theta_t == [0.0, 0.0, 0.0, 0.0, 0.0]
    @test policy.model.Sigma_t == Diagonal(ones(sum(labeling)))
    X = [1.0, 2.0, 5.0]
    W = 1
    Y = 1.0
    ContextualBandits.state_update!(policy,W,X,Y)
    @test policy.model.theta_t == [0.14285714285714285, 0.14285714285714285, 0.2857142857142857, 0.0, 0.0]
    rng = MersenneTwister(1234)
    w = ContextualBandits.allocation(policy,X,W,X,Y,rng)
    @test w == 1
    @test ContextualBandits.implementation(policy,X,W,X,Y) == [1]
    @test ContextualBandits.policy_labeling(policy) == labeling
    rng = MersenneTwister(1234)
    @test ContextualBandits.allocationIndependent(policy,X,[W],X,[Y],rng) == w
    @test ContextualBandits.implementationIndependent(policy,X,[W],X,[Y]) == [1]

    ContextualBandits.initialize!(policy,W,X,Y)
    @test policy.model.theta_t == [0.0, 2.142857142857143, 1.1428571428571428, 2.142857142857143, 2.0]
    @test policy.model.Sigma_t == [3.428571428571429 -0.5714285714285714 -1.1428571428571428 0.0 0.0; -0.5714285714285714 3.428571428571429 -1.1428571428571428 0.0 0.0; -1.1428571428571428 -1.1428571428571428 1.7142857142857144 0.0 0.0; 0.0 0.0 0.0 4.0 0.0; 0.0 0.0 0.0 0.0 4.0]
end

@testset "GreedyPolicyLinear" begin
    Wn = 2
    m = 3
    sample_std = 1.0
    labeling = [true, false, false, true, true, false, true, false, true]
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    policy = GreedyPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling)
    ContextualBandits.initialize!(policy,[],[],[])
    @test policy.model.theta_t == [0.0, 0.0, 0.0, 0.0, 0.0]
    @test policy.model.Sigma_t == Diagonal(ones(sum(labeling)))
    X = [1.0, 2.0, 5.0]
    W = [1]
    Y = [1.0]
    ContextualBandits.state_update!(policy,W,X,Y)
    @test policy.model.theta_t == [0.14285714285714285, 0.14285714285714285, 0.2857142857142857, 0.0, 0.0]
    rng = MersenneTwister(1234)
    @test ContextualBandits.allocation(policy,X,W,X,Y,rng) == 1
    @test ContextualBandits.implementation(policy,X,W,X,Y) == [1]
    @test ContextualBandits.policy_labeling(policy) == labeling
end