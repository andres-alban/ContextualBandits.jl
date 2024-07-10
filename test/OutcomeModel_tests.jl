using ContextualBandits
using Random
using LinearAlgebra
using Test

@testset "OutcomeLinear" begin
    n = 2
    m = 3
    labeling = [true, false, false, true, true, false, true, false, true]
    sample_std = 1.0
    mu = [1.0, 2.0, -1.0, -1.0, 2.0]
    outcome_model = OutcomeLinear(n, m, mu, sample_std, labeling)
    rng = MersenneTwister(1234)
    ContextualBandits.outcome_model_state!(outcome_model,rng)
    X = [1.0, 2.0, 5.0]
    W = 1
    @test ContextualBandits.mean_outcome(outcome_model,W,X) == 1.0
    Z = ContextualBandits.noise_outcome(outcome_model,rng)
    @test Z == 0.8673472019512456
    @test ContextualBandits.noisy_outcome(outcome_model,W,X,Z) == 1.8673472019512456
end

@testset "OutcomeLinearBayes" begin
    n = 2
    m = 3
    labeling = [true, false, false, true, true, false, true, false, true]
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    outcome_model = OutcomeLinearBayes(n, m, theta0, Sigma0, sample_std, labeling)
    rng = MersenneTwister(1234)
    ContextualBandits.outcome_model_state!(outcome_model,rng)
    @test outcome_model.mu == [0.8673472019512456, -0.9017438158568171, -0.4944787535042339, -0.9029142938652416, 0.8644013132535154]
    X = [1.0, 2.0, 5.0]
    W = 1
    @test ContextualBandits.mean_outcome(outcome_model,W,X) == -1.0233541209140393
    Z = ContextualBandits.noise_outcome(outcome_model,rng)
    @test Z == 2.2118774995743475
    @test ContextualBandits.noisy_outcome(outcome_model,W,X,Z) == 1.1885233786603082
end

@testset "OutcomeLinearBayes semidefinite covariance matrix" begin
    n = 2
    m = 3
    labeling = [true, false, false, true, true, false, true, false, true]
    theta0 = zeros(sum(labeling))
    Sigma0 = ones(sum(labeling),sum(labeling))
    sample_std = 1.0
    outcome_model = OutcomeLinearBayes(n, m, theta0, Sigma0, sample_std, labeling)
    rng = MersenneTwister(1234)
    ContextualBandits.outcome_model_state!(outcome_model,rng)
    @test outcome_model.mu ==  [0.8673472019512456, 0.8673472019512456, 0.8673472019512456, 0.8673472019512456, 0.8673472019512456]
    X = [1.0, 2.0, 5.0]
    W = 1
    @test ContextualBandits.mean_outcome(outcome_model,W,X) == 3.4693888078049824
    Z = ContextualBandits.noise_outcome(outcome_model,rng)
    @test Z == -0.9017438158568171
    @test ContextualBandits.noisy_outcome(outcome_model,W,X,Z) == 2.5676449919481654
end