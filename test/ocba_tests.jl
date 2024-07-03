using ContextualBandits
using Random
using Distributions
using LinearAlgebra
using Test

@testset "OCBA policy" begin
    Wn = 3
    FX = CovariatesIndependent([Categorical(1/2,1/2), Uniform(0,1)])
    m = length(FX)
    labeling=[false, false, true, true, true, false, true, true, false, true, true, false]
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    predictive = [2]
    policy = OCBAPolicyLinear(Wn, m, theta0, Sigma0, sample_std, predictive, labeling)
    ContextualBandits.initialize!(policy)
    @test policy.model.theta_t == theta0
    @test policy.model.Sigma_t == Sigma0
    rng = MersenneTwister(1234)
    Xcurrent = rand(rng, FX)
    X = rand(rng, FX, 20)
    W = rand(rng, 1:Wn, 20)
    Y = randn(rng, 20)
    for i in eachindex(Y)
        ContextualBandits.state_update!(policy, W[1:i], view(X,:,1:i), Y[1:i])
    end
    w = ContextualBandits.allocation(policy, Xcurrent, W, X, Y, rng)
    @test w == 1
    ContextualBandits.state_update!(policy, w, Xcurrent, randn(rng))
    @test policy.model.theta_t == [-0.44985531048383354, 0.07975310581992098, 0.5199533257605635, 0.00019141649694241758, 0.47741555990786033, -0.22161555149193185, -0.03742294853375851]
    @test ContextualBandits.implementation(policy, Xcurrent, W, X, Y) == [1]
end
