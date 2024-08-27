using ContextualBandits
using Random
using Distributions
using BenchmarkTools
using Test

@testset "LassoCVLabelingSelector" begin
    samplesize = 6000
    rng = Xoshiro(122)
    FX = CovariatesIndependent([Normal(), Normal()]) # Covariates with an intercept and two normally distributed covriates 
    X = rand(rng, FX, samplesize)
    n = 2
    W = rand(rng, 1:n, samplesize)
    labeling = ones(Bool, (n + 1) * length(FX))
    labeling[1] = 0
    # covariate matrix
    WX = interact(W, n, X, labeling)
    sigma = 0.1
    theta_true = [0, 1, 0, 1, -1, 0, 0, 2, 1]
    # outcome variable
    y = vcat(ones(size(WX, 2))', WX)' * theta_true + sigma * randn(rng, samplesize)

    selector = LassoCVLabelingSelector(n, length(FX))

    newlabeling = ContextualBandits.labeling_selection(selector, W, X, y)
    @test length(newlabeling) == length(labeling)
    @test sum(newlabeling) < length(newlabeling)
end