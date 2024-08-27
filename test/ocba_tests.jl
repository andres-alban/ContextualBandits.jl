using ContextualBandits
using Random
using Distributions
using LinearAlgebra
using Test

@testset "OCBA policy" begin
    n = 3
    FX = CovariatesIndependent([Categorical(1/2,1/2), Uniform(0,1)])
    m = length(FX)
    labeling=[false, false, true, true, true, false, true, true, false, true, true, false]
    theta0 = zeros(sum(labeling))
    Sigma0 = diagm(ones(sum(labeling)))
    sample_std = 1.0
    predictive = [2]
    policy = OCBAPolicyLinear(n, m, theta0, Sigma0, sample_std, predictive, labeling)
    ContextualBandits.initialize!(policy)
    @test policy.model.theta_t == theta0
    @test policy.model.Sigma_t == Sigma0
    rng = Xoshiro(1234)
    Xcurrent = rand(rng, FX)
    X = rand(rng, FX, 20)
    W = rand(rng, 1:n, 20)
    Y = randn(rng, 20)
    for i in eachindex(Y)
        ContextualBandits.state_update!(policy, W[1:i], view(X,:,1:i), Y[1:i])
    end
    w = ContextualBandits.allocation(policy, Xcurrent, W, X, Y, rng)
    @test w in 1:n
    Ycurrent = randn(rng)
    ContextualBandits.state_update!(policy, w, Xcurrent, Ycurrent, rng)
    @test policy.model.theta_t ≈ BayesUpdateNormal(theta0, Sigma0, interact([W;w], n, hcat(X,Xcurrent), labeling), [Y; Ycurrent], sample_std)[1]
    @test all(ContextualBandits.implementation(policy, Xcurrent, W, X, Y) .∈ Ref(1:n))
end
