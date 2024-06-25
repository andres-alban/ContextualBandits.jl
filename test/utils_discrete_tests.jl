using ContextualBandits
using Distributions
using LinearAlgebra
using Test

@testset "X2g and g2X " begin
    FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),OrdinalDiscrete([1/3,1/3,1/3])])
    X = [1.0, 0.0, 0.0, 0.0]
    g = ContextualBandits.X2g(X,FX)
    @test g == 1
    @test ContextualBandits.g2X(g,FX) == X
    X = [1.0, 0.0, 1.0, 2.0]
    g = ContextualBandits.X2g(X,FX)
    @test g == 9
    @test ContextualBandits.g2X(g,FX) == X
    X = [1.0, 1.0, 0.0, 1.0]
    g = ContextualBandits.X2g(X,FX)
    @test g == 5
    @test ContextualBandits.g2X(g,FX) == X
end

@testset "X2g_prior" begin
    FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),OrdinalDiscrete([1/3,1/3,1/3])])
    labels = ones(Bool, 16)
    Wn = 3
    gn = ContextualBandits.total_groups(FX)
    gs = zeros(Int, gn*Wn)
    ws = zeros(Int, gn*Wn)
    for i in eachindex(gs)
        ws[i] = ContextualBandits.index2treatment(i,gn)
        gs[i] = ContextualBandits.index2g(i,gn)
    end
    @test ws == vcat(ones(Int,gn),2*ones(Int,gn),3*ones(Int,gn))
    @test gs == repeat(1:9,3)

    theta0 = rand((Wn+1)*length(FX))
    Sigma0 = rand(16,16)
    Sigma0 = Sigma0 * Sigma0'
    theta0_disc, Sigma0_disc = ContextualBandits.X2g_prior(theta0,Sigma0,FX,labels,Wn)
    for i in eachindex(theta0_disc)
        w = ContextualBandits.index2treatment(i,gn)
        g = ContextualBandits.index2g(i,gn)
        X = ContextualBandits.g2X(g,FX)
        @test theta0_disc[i] ≈ interact(w,Wn,X,labels)' * theta0
        @test Sigma0_disc[i,i] ≈ interact(w,Wn,X,labels)' * Sigma0 * interact(w,Wn,X,labels)
        for j in eachindex(theta0_disc)
            wj = ContextualBandits.index2treatment(j,gn)
            gj = ContextualBandits.index2g(j,gn)
            Xj = ContextualBandits.g2X(gj,FX)
            @test Sigma0_disc[i,j] ≈ interact(w,Wn,X,labels)' * Sigma0 * interact(wj,Wn,Xj,labels)
        end
    end
end

@testset "X2g_probs" begin
    FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),OrdinalDiscrete([1/3,1/3,1/3])])
    p = ContextualBandits.X2g_probs(FX)
    @test p ≈ [1/9 for i in 1:9]
    FX = CovariatesCopula([Categorical([1/3,1/3,1/3]),OrdinalDiscrete([1/3,1/3,1/3])],0.3)
    p = ContextualBandits.X2g_probs(FX)
    @test sum(p) == 1.0
end