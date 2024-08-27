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
    rng = Xoshiro(1234)
    ContextualBandits.outcome_model_state!(outcome_model,rng)
    X = [1.0, 2.0, 5.0]
    W = 1
    @test ContextualBandits.mean_outcome(outcome_model,W,X) == 1.0
    Z = ContextualBandits.noise_outcome(outcome_model,rng)
    @test Z < 10 && Z > -10
    @test ContextualBandits.noisy_outcome(outcome_model,W,X,Z) == 1 + Z
end

@testset "OutcomeLinearBayes" begin
    n = 2
    m = 3
    labeling = [true, false, false, true, true, false, true, false, true]
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    outcome_model = OutcomeLinearBayes(n, m, theta0, Sigma0, sample_std, labeling)
    rng = Xoshiro(1234)
    ContextualBandits.outcome_model_state!(outcome_model,rng)
    @test !(outcome_model.mu ≈ outcome_model.theta0)
    X = [1.0, 2.0, 5.0]
    W = 1
    mean_out = ContextualBandits.mean_outcome(outcome_model,W,X)
    @test mean_out == interact(W, n, X, labeling)' * outcome_model.mu
    Z = ContextualBandits.noise_outcome(outcome_model,rng)
    @test ContextualBandits.noisy_outcome(outcome_model,W,X,Z) == mean_out + Z
end

@testset "OutcomeLinearBayes semidefinite covariance matrix" begin
    n = 2
    m = 3
    labeling = [true, false, false, true, true, false, true, false, true]
    theta0 = zeros(sum(labeling))
    Sigma0 = ones(sum(labeling),sum(labeling))
    sample_std = 1.0
    outcome_model = OutcomeLinearBayes(n, m, theta0, Sigma0, sample_std, labeling)
    rng = Xoshiro(1234)
    ContextualBandits.outcome_model_state!(outcome_model,rng)
    @test all(outcome_model.mu .== outcome_model.mu[1])
    X = [1.0, 2.0, 5.0]
    W = 1
    mean_out = ContextualBandits.mean_outcome(outcome_model,W,X)
    @test mean_out == interact(W, n, X, labeling)' * outcome_model.mu
    Z = ContextualBandits.noise_outcome(outcome_model,rng)
    @test ContextualBandits.noisy_outcome(outcome_model,W,X,Z) == mean_out + Z
end