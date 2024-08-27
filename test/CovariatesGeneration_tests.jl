using ContextualBandits
using Distributions
using Copulas
using Random
using BenchmarkTools
using Test

@testset "CovariatesCopula and CovariatesIndependent" begin
    copula = GaussianCopula([1 0.0; 0.0 1])
    FX = CovariatesCopula([Categorical([1 / 3, 1 / 3, 1 / 3]), Normal(0, 1)], copula)
    FXi = CovariatesIndependent([Categorical([1 / 3, 1 / 3, 1 / 3]), Normal(0, 1)])

    rng = Xoshiro(1234)
    X = rand(rng, FX)
    @test X[1] == 1.0
    @test sum(X[2:3]) == 1.0 || sum(X[2:3]) == 0.0
    @test all(X[2:3] .∈ Ref([0, 1]))
    @test X[4] < 10 && X[4] > -10 && X[4] != 0.0

    rng = Xoshiro(1234)
    Xi = rand(rng, FXi)
    @test Xi[1] == 1.0
    @test sum(Xi[2:3]) == 1.0 || sum(Xi[2:3]) == 0.0
    @test all(Xi[2:3] .∈ Ref([0, 1]))
    @test Xi[4] < 10 && Xi[4] > -10 && X[4] != 0.0

    # CovariatesIndependent is slightly faster than CovariatesCopula
    # @benchmark rand($rng,$FX)
    # @benchmark rand($rng,$FXi)
end

@testset "CovariatesInteracted" begin
    rng = Xoshiro(1234)
    copula = GaussianCopula([1 0.2; 0.2 1])
    FX = CovariatesCopula([Categorical([1 / 3, 1 / 3, 1 / 3]), Normal(0, 1)], copula)
    FXinteracted = CovariatesInteracted(FX, [x -> x[1], x -> x[2], x -> x[3], x -> x[4], x -> x[2] * x[4], x -> x[3] * x[4]])
    X = rand(rng, FXinteracted)
    @test length(X) == 6
    @test X[1] == 1.0
    @test sum(X[2:3]) == 1.0 || sum(X[2:3]) == 0.0
    @test all(X[2:3] .∈ Ref([0, 1]))
    @test X[4] < 10 && X[4] > -10 && X[4] != 0.0
    @test X[5] .∈ Ref([0, X[4]])
    @test X[6] .∈ Ref([0, X[4]])
end

@testset "OrdinalDiscrete" begin
    rng = Xoshiro(1234)
    FX = CovariatesIndependent([OrdinalDiscrete([1 / 3, 1 / 3, 1 / 3]), Normal(0, 1)])
    X = rand(rng, FX)
    @test X[1] == 1.0
    @test X[2] in 0:2
    @test X[3] < 10 && X[3] > -10 && X[3] != 0.0
end

@testset "covariates_partition" begin
    FX = CovariatesIndependent([Categorical([1 / 3, 1 / 3, 1 / 3]), Normal(0, 1)])
    @test covariates_partition(FX) == [[2, 3], [4]]
    FX = CovariatesCopula([Categorical([1 / 3, 1 / 3, 1 / 3]), Normal(0, 1)], GaussianCopula([1 0.2; 0.2 1]), false)
    @test covariates_partition(FX) == [[1, 2, 3], [4]]
    FX = CovariatesCopula([Categorical([1 / 3, 1 / 3, 1 / 3]), Normal(0, 1)], GaussianCopula([1 0.2; 0.2 1]), true, false)
    @test covariates_partition(FX) == [[2, 3, 4], [5]]
    FX = CovariatesIndependent([Categorical([1 / 3, 1 / 3, 1 / 3]), Categorical(ones(4) / 4)])
    @test covariates_partition(FX) == [[2, 3], [4, 5, 6]]
    FX = CovariatesIndependent([Categorical([1 / 3, 1 / 3, 1 / 3]), Categorical([1])])
    @test covariates_partition(FX) == [[2, 3]]
    FX = CovariatesIndependent([Categorical([1 / 3, 1 / 3, 1 / 3]), Categorical([1])], false)
    @test covariates_partition(FX) == [[1, 2, 3], [4]]
end