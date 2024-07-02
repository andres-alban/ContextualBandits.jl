using ContextualBandits
using Distributions
using Copulas
using Random
using BenchmarkTools
using Test

@testset "CovariatesCopula" begin
    rng = MersenneTwister(1234)
    copula = GaussianCopula([1 0.2; 0.2 1])
    FX = CovariatesCopula([Categorical([1/3,1/3,1/3]),Normal(0,1)],copula)
    @test rand(rng,FX) == [1.0, 0.0, 1.0, -0.7100554506335256]
end

@testset "CovariatesIndependent" begin
    copula = GaussianCopula([1 0.0; 0.0 1])
    FX = CovariatesCopula([Categorical([1/3,1/3,1/3]),Normal(0,1)],copula)
    FXi = CovariatesIndependent([Categorical([1/3,1/3,1/3]),Normal(0,1)])

    rng = MersenneTwister(1234)
    @test rand(rng,FX) == [1.0, 1.0, 0.0, 0.7283392731742008]
    rng = MersenneTwister(1234)
    @test rand(rng,FXi) == [1.0, 1.0, 0.0, -0.9017438158568171]
    
    # CovariatesIndependent is slightly faster than CovariatesCopula
    # @benchmark rand($rng,$FX)
    # @benchmark rand($rng,$FXi)
end

@testset "CovariatesInteracted" begin
    rng = MersenneTwister(1234)
    copula = GaussianCopula([1 0.2; 0.2 1])
    FX = CovariatesCopula([Categorical([1/3,1/3,1/3]),Normal(0,1)],copula)
    FXinteracted = CovariatesInteracted(FX,[x->x[1], x->x[2], x->x[2]*x[4], x->x[3]*x[4]])
    @test rand(rng,FXinteracted) == [1.0, 0.0, 0.0, -0.7100554506335256]
end

@testset "OrdinalDiscrete" begin
    rng = MersenneTwister(1234)
    FX = CovariatesIndependent([OrdinalDiscrete([1/3,1/3,1/3]),Normal(0,1)])
    @test rand(rng,FX) ==[1.0, 1.0, -0.9017438158568171]
end

@testset "covariates_partition" begin
    FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),Normal(0,1)])
    @test covariates_partition(FX) == [[2,3],[4]]
    FX = CovariatesCopula([Categorical([1/3,1/3,1/3]),Normal(0,1)],GaussianCopula([1 0.2; 0.2 1]),false)
    @test covariates_partition(FX) == [[1,2,3],[4]]
    FX = CovariatesCopula([Categorical([1/3,1/3,1/3]),Normal(0,1)],GaussianCopula([1 0.2; 0.2 1]),true,false)
    @test covariates_partition(FX) == [[2,3,4],[5]]
    FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),Categorical(ones(4)/4)])
    @test covariates_partition(FX) == [[2,3],[4,5,6]]
    FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),Categorical([1])])
    @test covariates_partition(FX) == [[2,3]]
    FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),Categorical([1])],false)
    @test covariates_partition(FX) == [[1,2,3],[4]]
end