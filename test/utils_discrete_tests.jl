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
    Wn = 3
    labeling = ones(Bool, (Wn+1)*length(FX))
    gn = ContextualBandits.total_groups(FX)
    gs = zeros(Int, gn*Wn)
    ws = zeros(Int, gn*Wn)
    for i in eachindex(gs)
        ws[i] = ContextualBandits.index2treatment(i,gn)
        gs[i] = ContextualBandits.index2g(i,gn)
    end
    @test ws == vcat(ones(Int,gn),2*ones(Int,gn),3*ones(Int,gn))
    @test gs == repeat(1:9,3)
    @test ContextualBandits.treatment_g2index.(ws,gs,gn) == 1:length(gs)

    len = (Wn+1)*length(FX)
    theta0 = rand(len)
    Sigma0 = rand(len,len)
    Sigma0 = Sigma0 * Sigma0'
    theta0_disc, Sigma0_disc = ContextualBandits.X2g_prior(theta0,Sigma0,FX,labeling,Wn)
    theta_expect = similar(theta0_disc)
    Sigma_expect = similar(Sigma0_disc)
    for i in eachindex(theta0_disc)
        w = ContextualBandits.index2treatment(i,gn)
        g = ContextualBandits.index2g(i,gn)
        X = ContextualBandits.g2X(g,FX)
        theta_expect[i] = interact(w,Wn,X,labeling)' * theta0
        for j in eachindex(theta0_disc)
            wj = ContextualBandits.index2treatment(j,gn)
            gj = ContextualBandits.index2g(j,gn)
            Xj = ContextualBandits.g2X(gj,FX)
            Sigma_expect[i,j] = interact(w,Wn,X,labeling)' * Sigma0 * interact(wj,Wn,Xj,labeling)
        end
    end
    @test theta0_disc ≈ theta_expect
    @test Sigma0_disc ≈ Sigma_expect
end

@testset "X2g_probs" begin
    FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),OrdinalDiscrete([1/3,1/3,1/3])])
    p = ContextualBandits.X2g_probs(FX)
    @test p ≈ [1/9 for i in 1:9]
    FX = CovariatesCopula([Categorical([1/3,1/3,1/3]),OrdinalDiscrete([1/3,1/3,1/3])],0.3)
    p = ContextualBandits.X2g_probs(FX)
    @test sum(p) == 1.0
end

@testset "BayesUpdateNormalDiscrete" begin
    theta = [0.0, 0.0]
    Sigma = [1.0 0.0; 0.0 1.0]
    g = 1
    y = 1.0
    sample_std = 1.0
    expected_theta = [0.5, 0.0]
    expected_Sigma = [0.5 0.0; 0.0 1.0]
    @test ContextualBandits.BayesUpdateNormalDiscrete(theta, Sigma, g, y, sample_std) == (expected_theta, expected_Sigma)
    X = zeros(length(theta))
    X[g] = 1
    @test BayesUpdateNormal(theta, Sigma, X, y, sample_std) == (expected_theta, expected_Sigma)

    theta = [0.0, 0.0]
    Sigma = [1.0 0.0; 0.0 1.0]
    g = [1,2]
    y = [1.0,2.0]
    sample_std = 1.0
    expected_theta = [0.5, 1.0]
    expected_Sigma = [0.5 0.0; 0.0 0.5]
    @test ContextualBandits.BayesUpdateNormalDiscrete(theta, Sigma, g, y, sample_std) == (expected_theta, expected_Sigma)

    theta = [0.0, 0.0]
    Sigma = [1.0 0.0; 0.0 1.0]
    g = [1,2]
    y = [1.0,2.0]
    sample_std = [1.0,2.0]
    expected_theta = [0.5, 0.4]
    expected_Sigma = [0.5 0.0; 0.0 0.8]
    @test ContextualBandits.BayesUpdateNormalDiscrete(theta, Sigma, g, y, sample_std) == (expected_theta, expected_Sigma)

end