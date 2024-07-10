using ContextualBandits
using Distributions
using LinearAlgebra
using Random
using Test

@testset "KG functions" begin
    theta = rand(10)
    sigmasq = rand(10)
    sample_std = rand(10)
    Sigma = Diagonal(sigmasq)
    c = cKG(theta,Sigma,sample_std)
    i = iKG(theta,sigmasq,sample_std)
    @test c ≈ i

    sample_std = 0.1
    c = cKG(theta,Sigma,sample_std)
    i = iKG(theta,sigmasq,sample_std)
    @test c ≈ i

    rng = MersenneTwister(1234)
    theta = rand(rng,10)
    Sigma = rand(rng,10,10)
    Sigma = Sigma * Sigma'
    sample_std = rand(rng,10)
    c = cKG(theta,Sigma,sample_std)
    @test c == [-0.8846588839665859, -0.7741491951129047, -0.9596368128224648, -0.9792457475294534, -0.7426307677119657, -2.124034868230084, -1.4332461291574268, -1.2420862648959767, -1.977175841884483, -0.8928851843709824]
end