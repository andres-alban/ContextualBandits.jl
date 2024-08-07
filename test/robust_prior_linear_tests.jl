using ContextualBandits
using LinearAlgebra
using Test

@testset "prior_linear" begin
    n = 3
    m = 3
    sigma0 = 1.0
    psi = log(2)
    D = [0 1 2;
        1 0 1;
        2 1 0]
    labeling = BitVector([0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
    theta0, Sigma0 = default_prior_linear(n, m, sigma0, psi, D, labeling)
    @test theta0 == zeros(sum(labeling))
    @test Sigma0 == [
        1.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 0.0 0.5 0.0 0.25 0.0;
        0.0 0.0 1.0 0.0 0.5 0.0 0.25;
        0.0 0.5 0.0 1.0 0.0 0.5 0.0;
        0.0 0.0 0.5 0.0 1.0 0.0 0.5;
        0.0 0.25 0.0 0.5 0.0 1.0 0.0;
        0.0 0.0 0.25 0.0 0.5 0.0 1.0]

    robustify_prior_linear!(theta0, Sigma0, n, m, labeling)
    @test theta0 == [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    @test Sigma0 == [4.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 4.0 0.0 2.0 0.0 1.0 0.0;
        0.0 0.0 4.0 0.0 2.0 0.0 1.0;
        0.0 2.0 0.0 4.0 0.0 2.0 0.0;
        0.0 0.0 2.0 0.0 4.0 0.0 2.0;
        0.0 1.0 0.0 2.0 0.0 4.0 0.0;
        0.0 0.0 1.0 0.0 2.0 0.0 4.0]

    theta0, Sigma0 = default_prior_linear(n, m, sigma0, psi)
    @test theta0 == zeros(n * m)
    @test Sigma0 == Diagonal(sigma0 * ones(n * m))
end
