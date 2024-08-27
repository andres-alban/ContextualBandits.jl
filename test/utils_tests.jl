using ContextualBandits
using Random
using Distributions
using Test

# Test interact function
@testset "interact function" begin
    w = [1, 2]
    n = 2
    x = [1 1; 2 3; 5 4]
    labeling = Bool.([1, 1, 0, 0, 0, 1, 0, 1, 0])
    expected_output = [1.0 1.0; 2.0 3.0; 5.0 0.0; 0.0 3.0]
    @test interact(w, n, x, labeling) == expected_output

    w = w[1]
    x = x[:, 1]
    expected_output = [1.0, 2.0, 5.0, 0.0]
    @test interact(w, n, x, labeling) == expected_output

    # Test interact with default labeling
    w = [1, 2]
    n = 2
    x = [1 1; 2 3; 5 4]
    expected_output =
        [1.0 0.0;
            2.0 0.0;
            5.0 0.0;
            0.0 1.0;
            0.0 3.0;
            0.0 4.0]
    @test interact(w, n, x) == expected_output

    WX = Matrix{Float64}(undef, 6, 2)
    interact!(WX, w, n, x)
    @test WX == expected_output

    w = w[1]
    x = x[:, 1]
    expected_output = expected_output[:, 1]
    @test interact(w, n, x) == expected_output
end

# Test BayesUpdateNormal function
@testset "BayesUpdateNormal function" begin
    theta = [0.0, 0.0]
    Sigma = [1.0 0.0; 0.0 1.0]
    X = [1.0, 2.0]
    y = 1.0
    sample_std = 1.0
    expected_theta = [0.16666666666666666, 0.3333333333333333]
    expected_Sigma = [0.8333333333333334 -0.3333333333333333; -0.3333333333333333 0.33333333333333337]
    @test BayesUpdateNormal(theta, Sigma, X, y, sample_std) == (expected_theta, expected_Sigma)

    theta = [0.0, 0.0]
    Sigma = [1.0 0.0; 0.0 1.0]
    X = [1.0 2.0 3.0; 4.0 5.0 6.0]
    y = [1.0, 2.0, 3.0]
    sample_std = 1.0
    expected_theta = [0.4657534246575341, 0.2191780821917809]
    expected_Sigma = [0.5342465753424659 -0.21917808219178087; -0.21917808219178087 0.1027397260273973]
    @test BayesUpdateNormal(theta, Sigma, X, y, sample_std) == (expected_theta, expected_Sigma)

    theta = [0.0, 0.0]
    Sigma = [1.0 0.0; 0.0 1.0]
    X = [1.0 2.0 3.0; 4.0 5.0 6.0]
    y = [1.0, 2.0, 3.0]
    sample_std = [1.0, 2.0, 3.0]
    expected_theta = [0.25850340136054417, 0.23129251700680276]
    expected_Sigma = [0.7414965986394558 -0.23129251700680276; -0.23129251700680276 0.10884353741496602]
    @test BayesUpdateNormal(theta, Sigma, X, y, sample_std) == (expected_theta, expected_Sigma)
end

# Test argmax_ties function
@testset "argmax_ties and argmin_ties functions" begin
    itr = [1, 2, 3, 4, 5]
    @test argmax_ties(itr) == 5
    @test argmin_ties(itr) == 1

    itr = [1, 2, 3, 3, 2, 1]
    rng = Xoshiro(1234)
    @test argmax_ties(itr, rng) in [3, 4]
    @test argmin_ties(itr, rng) in [1, 6]
end

@testset "randnMv" begin
    mu = [0.0, 0.0]
    Sigma = [1.0 0.0; 0.0 1.0]
    rng = Xoshiro(1234)
    rng2 = Xoshiro(1234)
    @test randnMv(rng, mu, Sigma) == rand(rng2, MvNormal(mu, Sigma))
    Sigma = ones(2, 2)
    x = randnMv(Random.default_rng(), mu, Sigma)
    @test x[1] == x[2]
end

@testset "labeling2predprog" begin
    n = 3
    m = 3
    labeling = Bool[0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    pred, prog = labeling2predprog(n, m, labeling)
    @test pred == [2]
    @test prog == [[3]]

    labeling = Bool[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pred, prog = labeling2predprog(n, m, labeling)
    @test pred == []
    @test prog == [[2], [3]]

    labeling = Bool[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    pred, prog = labeling2predprog(n, m, labeling)
    @test pred == [2, 3]
    @test prog == []

    n = 2
    m = 5
    partition = [[2, 3, 4], [5]]
    labeling = Bool[
        0, 1, 1, 1, 1,
        1, 1, 0, 0, 0,
        1, 0, 0, 0, 0]
    pred, prog = labeling2predprog(n, m, labeling, partition)
    @test pred == [2]
    @test prog == [[3, 4], [5]]
    FX = CovariatesIndependent([Categorical(ones(4) / 4), Normal()])
    @test partition == covariates_partition(FX)
    pred, prog = labeling2predprog(n, FX, labeling)
    @test pred == [2]
    @test prog == [[3, 4], [5]]


    FX = CovariatesIndependent([Categorical(ones(4) / 4), Normal()], false)
    pred, prog = labeling2predprog(n, FX, labeling)
    @test pred == [1, 2]
    @test prog == [[3, 4], [5]]
end