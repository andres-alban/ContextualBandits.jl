using ContextualBandits
using Random
using LinearAlgebra
using Test

@testset "Thompson Sampling policy" begin
    n = 3
    m = 3
    labeling = vcat(falses(m), trues(n * m))
    theta0 = zeros(sum(labeling))
    Sigma0 = diagm(ones(sum(labeling)))
    sample_std = 1.0
    policy = TSPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    ContextualBandits.initialize!(policy)
    @test policy.model.theta_t == theta0
    @test policy.model.Sigma_t == Sigma0
    rng = Xoshiro(1234)
    Xcurrent = [1.0, 2.0, 5.0]
    w = ContextualBandits.allocation(policy, Xcurrent, [], [], [], rng)
    @test w in 1:n
    ContextualBandits.state_update!(policy, w, Xcurrent, 1.0, rng)
    @test policy.model.theta_t ≈ BayesUpdateNormal(theta0, Sigma0, interact(w, n, Xcurrent, labeling), 1.0, sample_std)[1]
    @test all(ContextualBandits.implementation(policy, Xcurrent, [], [], []) .∈ Ref(1:n))
end

@testset "Top Two Thompson Sampling policy" begin
    n = 3
    m = 3
    labeling = vcat(falses(m), trues(n * m))
    theta0 = zeros(sum(labeling))
    Sigma0 = diagm(ones(sum(labeling)))
    sample_std = 1.0
    beta = 0.5
    maxiter = 100
    policy = TTTSPolicyLinear(n, m, theta0, Sigma0, sample_std, beta, maxiter, labeling)
    ContextualBandits.initialize!(policy)
    @test policy.model.theta_t == theta0
    @test policy.model.Sigma_t == Sigma0
    rng = Xoshiro(1234)
    Xcurrent = [1.0, 2.0, 5.0]
    w = ContextualBandits.allocation(policy, Xcurrent, [], [], [], rng)
    @test w in 1:n
    ContextualBandits.state_update!(policy, w, Xcurrent, 1.0, rng)
    @test policy.model.theta_t ≈ BayesUpdateNormal(theta0, Sigma0, interact(w, n, Xcurrent, labeling), 1.0, sample_std)[1]
    @test all(ContextualBandits.implementation(policy, Xcurrent, [], [], []) .∈ Ref(1:n))
end