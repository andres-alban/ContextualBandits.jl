using ContextualBandits
using Random
using Distributions
using Test

@testset "fEVI_MC family functions" begin
    rng = Xoshiro(1234)
    n = 3
    FXtilde = CovariatesIndependent([Categorical(1/3,1/3,1/3), OrdinalDiscrete([1/2,1/2])])
    m = length(FXtilde)
    labeling = vcat(falses(m),trues(n*m))
    theta = rand(rng,sum(labeling))
    Sigma = rand(rng,sum(labeling),sum(labeling))
    Sigma = Sigma * Sigma'
    sample_std = 1.0
    Xt = rand(rng,FXtilde)
    Wpipeline = Int[]
    Xpipeline = Float64[]
    etaon = 100
    etaoff = 100
    fEVI_MC_index = fEVI_MC(n, m, theta, Sigma, sample_std, Xt, FXtilde, Wpipeline, Xpipeline, etaon, etaoff, labeling, rng)
    fEVI_MC_index_indep = ContextualBandits.fEVI_MC_indep(n, m, theta, Sigma, sample_std, Xt, FXtilde, Wpipeline, Xpipeline, etaon, etaoff, labeling, rng)

    gn = ContextualBandits.total_groups(FXtilde)
    theta_disc, Sigma_disc = ContextualBandits.X2g_prior(theta, Sigma, FXtilde, labeling, n)
    gt = ContextualBandits.X2g(Xt, FXtilde)
    p = ContextualBandits.X2g_probs(FXtilde)
    fEVI_index = fEVI(n, gn, theta_disc, Sigma_disc, sample_std, gt, p)


    @test isapprox(fEVI_MC_index, fEVI_index, rtol=0.01)
    @test isapprox(fEVI_MC_index_indep, fEVI_index, rtol = 0.01)

    N = 100
    fEVI_MC_index_without_h = Matrix{Float64}(undef,n,N)
    fEVI_MC_index_without_h_indep = Matrix{Float64}(undef,n,N)
    for i in 1:N
        fEVI_MC_index_without_h[:,i] = ContextualBandits.fEVI_MC_without_h(n, m, theta, Sigma, sample_std, Xt, FXtilde, Wpipeline, Xpipeline, etaon, etaoff, labeling, rng)
        fEVI_MC_index_without_h_indep[:,i] = ContextualBandits.fEVI_MC_without_h_indep(n, m, theta, Sigma, sample_std, Xt, FXtilde, Wpipeline, Xpipeline, etaon, etaoff, labeling, rng)
    end
    @test all(mean(fEVI_MC_index_without_h,dims=2) + 2*std(fEVI_MC_index_without_h,dims=2)/sqrt(N) .> exp.(fEVI_index))
    @test all(mean(fEVI_MC_index_without_h,dims=2) - 2*std(fEVI_MC_index_without_h,dims=2)/sqrt(N) .< exp.(fEVI_index))
end

@testset "fEVI_MC family policies" begin
    rng = Xoshiro(1234)
    n = 3
    FXtilde = CovariatesIndependent([Categorical(1/3,1/3,1/3), Normal()])
    m = length(FXtilde)
    labeling = vcat(falses(m),trues(n*m))
    theta0 = rand(rng,sum(labeling))
    Sigma0 = rand(rng,sum(labeling),sum(labeling))
    Sigma0 = Sigma0 * Sigma0'
    sample_std = 1.0
    etaon = 10
    etaoff = 10
    policy = fEVI_MC_PolicyLinear(n, m, theta0, Sigma0, sample_std, FXtilde, etaon, etaoff, labeling)
    ContextualBandits.initialize!(policy)
    W = [1]
    X = rand(rng,FXtilde,1)
    Y = rand(rng,1)
    Xt = rand(rng,FXtilde)
    @test ContextualBandits.allocation(policy, Xt, W, X, Y, rng) in 1:n 
    ContextualBandits.state_update!(policy, W, X, Y, rng)
    @test ContextualBandits.allocation(policy, Xt, W, X, Y, rng) in 1:n
    @test all(ContextualBandits.implementation(policy, Xt, W, X, Y) .∈ Ref(1:n))
end