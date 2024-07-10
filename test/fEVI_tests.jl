using ContextualBandits
using Distributions
using Random
using Test

@testset "fEVI functions" begin
    n = 3
    gn = 3
    rng = MersenneTwister(1234)
    theta = rand(rng, n*gn)
    Sigma = rand(rng, n*gn, n*gn)
    Sigma = Sigma * Sigma'
    sample_std = rand(rng, n*gn)
    gt = rand(rng,1:gn)
    p = rand(rng,gn)
    p /= sum(p)
    fEVI_off = fEVI(n, gn, theta, Sigma, sample_std, gt, p)
    # The expected results were computed from other code to validate
    @test fEVI_off == [-2.289241727007539, -2.5010572326520846, -2.1910002668147732]

    rng = MersenneTwister(12345)
    theta = rand(rng, n*gn)
    Sigma = rand(rng, n*gn, n*gn)
    Sigma = Sigma * Sigma'
    sample_std = rand(rng, n*gn)
    gt = rand(rng,1:gn)
    p = rand(rng,gn)
    p /= sum(p)
    fEVI_off = fEVI(n, gn, theta, Sigma, sample_std, gt, p)
    # The expected results were computed from other code to validate
    @test fEVI_off == [-1.8715300384364837, -2.829324939335114, -2.4135617711801074]
end

@testset "fEVI policies" begin
    rng = MersenneTwister(1234)
    n = 3
    FX = CovariatesIndependent([Categorical([1/4,1/2,1/4]), OrdinalDiscrete([1/3,1/3,1/3])])
    m = length(FX)
    labeling = [true, false, false, true, 
                true, true, true, false,
                true, true, true, false,
                false, true, true, false]
    theta0 = rand(rng, sum(labeling))
    Sigma0 = rand(rng, sum(labeling), sum(labeling))
    Sigma0 = Sigma0 * Sigma0'
    sample_std = rand(rng)
    fEVI_off = fEVIDiscrete(n, m, theta0, Sigma0, sample_std, FX, labeling)
    P = 0
    T = 10
    fEVI_on = fEVIDiscreteOnOff(n, m, theta0, Sigma0, sample_std, FX, P, T, labeling)
    ContextualBandits.initialize!(fEVI_off)
    ContextualBandits.initialize!(fEVI_on)
    W = 1
    X = rand(rng, FX)
    Y = rand(rng)
    ContextualBandits.state_update!(fEVI_off, W, X, Y)
    ContextualBandits.state_update!(fEVI_on, W, X, Y)
    Xcurrent = rand(rng, FX)
    w = ContextualBandits.allocation(fEVI_off, Xcurrent, W, X, Y, rng)
    @test w == 2
    w = ContextualBandits.allocation(fEVI_on, Xcurrent, W, X, Y, rng)
    @test w == 2
    wpost = ContextualBandits.implementation(fEVI_off, Xcurrent, W, X, Y)
    @test wpost == [2]
    wpost = ContextualBandits.implementation(fEVI_on, Xcurrent, W, X, Y)
    @test wpost == [2]

    # The following tests that the policies can be simulated without errors
    policies = Dict("fEVIoff"=>fEVI_off, "fEVIon" => fEVI_on)
    mu = rand(rng, sum(labeling))
    outcome_model = OutcomeLinear(n, m, mu, sample_std, labeling)
    r = simulation_stochastic(10, FX, n, T, policies, outcome_model; post_reps=10, rng)
    r = r["output"]
end