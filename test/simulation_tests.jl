using ContextualBandits
using LinearAlgebra
using Random
using Distributed
using Distributions
using Test

@testset "finite_groups" begin
    FX = CovariatesIndependent([Normal(),Normal()])
    flag, weights, X_post = ContextualBandits.finite_groups(FX,20)
    @test flag == false && all(weights .== 1/20) && size(X_post) == (3,20)
    FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),Categorical([1/2,1/2])])
    flag, weights, X_post = ContextualBandits.finite_groups(FX,20)
    @test flag == true && all(weights .== 1/6) && X_post == [1.0 1.0 1.0 1.0 1.0 1.0; 0.0 0.0 1.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0 0.0 1.0]
    flag, weights, X_post = ContextualBandits.finite_groups(FX,5)
    @test flag == false && all(weights .== 1/5) && size(X_post) == (4,5)
end

@testset "simulation_replication StandardRecorder" begin
    T = 10
    m = 10
    Wn = 5
    X = rand(m, T)
    X_post = rand(m, 4)
    Z = rand(T)
    delay = 0
    labels = ones(Bool, (Wn+1)*m)
    theta0 = zeros(sum(labels))
    Sigma0 = Diagonal(ones(sum(labels)))
    sample_std = 1.0
    policy = RandomPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)
    ContextualBandits.initialize!(policy, [],[],[])
    outcome_model = LinearOutcomeLabelRandom(Wn, m, labels, theta0, Sigma0, sample_std)
    ContextualBandits.outcome_model_state!(outcome_model)
    Xinterest = X[:,3:4]
    X_post_weights = ones(4)./4
    recorder = ContextualBandits.StandardRecorder()
    ContextualBandits.initialize!(recorder, T, Wn, size(X,1), delay, size(X_post,2), X_post_weights, size(Xinterest,2))
    ContextualBandits.reset!(recorder,outcome_model,X_post,Xinterest)
    rng = MersenneTwister(1234)
    x = ContextualBandits.replication_stochastic(X,X_post,Z,Wn,delay,policy,outcome_model,recorder;Xinterest=Xinterest,X_post_weights=X_post_weights,rng=rng)
    names = ContextualBandits.output_recorder_names(recorder)
    @test all(x[findfirst(names .== "EOC_on")] .>= 0.0)
    @test all(cumsum(x[findfirst(names .== "EOC_on")]) .≈ x[findfirst(names .== "cumulEOC_on")])
    @test all(x[findfirst(names .== "PICS_on")] .∈ Ref([0,1]))
    @test all(x[findfirst(names .== "PICS_on")][x[findfirst(names .== "EOC_on")] .== 0.0] .== 0)
    @test all(cumsum(x[findfirst(names .== "PICS_on")]) .== x[findfirst(names .== "cumulPICS_on")])
    @test all(sum(x[findfirst(names .== "Wfrac_on")],dims=2) .== 1)
    @test all(x[findfirst(names .== "EOC_off")] .>= 0.0)
    @test all(0 .<= x[findfirst(names .== "PICS_off")] .<= 1)
    @test all(x[findfirst(names .== "PICS_off")][x[findfirst(names .== "EOC_off")] .== 0.0] .== 0)
    @test all(sum(x[findfirst(names .== "Wfrac_off")],dims=2) .== 1)
    @test all((x[findfirst(names .== "XEOC_on")] .>= 0.0) .| isnan.(x[findfirst(names .== "XEOC_on")]))
    @test all((x[findfirst(names .== "XPICS_on")] .∈ Ref([0,1])) .| isnan.(x[findfirst(names .== "XPICS_on")]))
    @test all(x[findfirst(names .== "XPICS_on")][x[findfirst(names .== "XEOC_on")] .== 0.0] .== 0)
    @test all(x[findfirst(names .== "Xfrac_on")] .∈ Ref([0,1]))
    @test all(x[findfirst(names .== "XWfrac_on")] .∈ Ref([0,1]))
    @test all(x[findfirst(names .== "XEOC_off")] .>= 0.0)
    @test all(x[findfirst(names .== "XPICS_off")] .∈ Ref([0,1]))
    @test all(x[findfirst(names .== "XPICS_off")][x[findfirst(names .== "XEOC_off")] .== 0.0] .== 0)
    @test all(x[findfirst(names .== "XWfrac_off")] .∈ Ref([0,1]))
    @test all(sum(x[findfirst(names .== "XWfrac_off")],dims=2) .== 1)
    @test all(x[findfirst(names .== "labelsfrac")] .== labels)
    @test all(x[findfirst(names .== "Nactivelabels")] .== sum(labels))
end

@testset "simulation_stochastic_internal" begin
    FX = CovariatesIndependent([Normal(),Normal()])
    FXtilde = FX
    Wn = 3
    m = length(FX)
    T = 10
    delay = 0
    labels = ones(Bool, (Wn+1)*m)
    theta0 = zeros(sum(labels))
    Sigma0 = Diagonal(ones(sum(labels)))
    sample_std = 1.0
    policy = RandomPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)
    policy2 = GreedyPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)
    policies = [policy, policy2]
    outcome_model = LinearOutcomeLabelRandom(Wn, m, labels, theta0, Sigma0, sample_std)
    reps = 10
    post_reps = 10
    pilot_samples_per_treatment = 0
    Xinterest = rand(FXtilde,2)
    rng = MersenneTwister(1234)
    x = ContextualBandits.simulation_stochastic_internal(FX,FXtilde,Wn,T,delay,policies,outcome_model;
        reps=reps,post_reps=post_reps, pilot_samples_per_treatment = pilot_samples_per_treatment,Xinterest=Xinterest,rng=rng)
    y = ContextualBandits.asdict(x[2])
    @test all(y["EOC_on"]["mean"] .>= 0.0)
    @test all(cumsum(y["EOC_on"]["mean"]) .≈ y["cumulEOC_on"]["mean"])
    @test all(0 .<= y["PICS_on"]["mean"] .<= 1)
    @test all(y["PICS_on"]["mean"][y["EOC_on"]["mean"] .== 0.0] .== 0)
    @test all(cumsum(y["PICS_on"]["mean"]) .≈ y["cumulPICS_on"]["mean"])
    @test all(sum(y["Wfrac_on"]["mean"],dims=2) .≈ 1)
    @test all(y["EOC_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["PICS_off"]["mean"] .<= 1)
    @test all(y["PICS_off"]["mean"][y["EOC_off"]["mean"] .== 0.0] .== 0)
    @test all(sum(y["Wfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["XEOC_on"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_on"]["mean"] .<= 1)
    @test all(y["XPICS_on"]["mean"][y["XEOC_on"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["Xfrac_on"]["mean"] .<= 1)
    @test all(0 .<= y["XWfrac_on"]["mean"] .<= 1)
    @test all(y["XEOC_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_off"]["mean"] .<= 1)
    @test all(y["XPICS_off"]["mean"][y["XEOC_off"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["XWfrac_off"]["mean"] .<= 1)
    @test all(sum(y["XWfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["labelsfrac"]["mean"] .== labels)
    @test all(y["Nactivelabels"]["mean"] .== sum(labels))
end

@testset "simulation_stochastic" begin
    FX = CovariatesCopula([Categorical([1/2,1/2]),OrdinalDiscrete([1/3,1/3,1/3])],0.3)
    FXtilde = FX
    Wn = 3
    m = length(FX)
    T = 10
    delay = 0
    labels = ones(Bool, (Wn+1)*m)
    theta0 = zeros(sum(labels))
    Sigma0 = Diagonal(ones(sum(labels)))
    sample_std = 1.0
    policy = RandomPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)
    policy2 = GreedyPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)
    policies = Dict("random" => policy, "greedy" => policy2)
    outcome_model = LinearOutcomeLabelRandom(Wn, m, labels, theta0, Sigma0, sample_std)
    reps = 10
    post_reps = 10
    pilot_samples_per_treatment = 0
    Xinterest = rand(FXtilde,2)
    rng = MersenneTwister(1234)
    x = simulation_stochastic(FX,FXtilde,Wn,T,delay,policies,outcome_model;
        reps=reps,post_reps=post_reps, pilot_samples_per_treatment = pilot_samples_per_treatment,Xinterest=Xinterest,rng=rng,verbose=true)
    y = x["output"]["random"]
    @test all(y["EOC_on"]["mean"] .>= 0.0)
    @test all(cumsum(y["EOC_on"]["mean"]) .≈ y["cumulEOC_on"]["mean"])
    @test all(0 .<= y["PICS_on"]["mean"] .<= 1)
    @test all(y["PICS_on"]["mean"][y["EOC_on"]["mean"] .== 0.0] .== 0)
    @test all(cumsum(y["PICS_on"]["mean"]) .≈ y["cumulPICS_on"]["mean"])
    @test all(sum(y["Wfrac_on"]["mean"],dims=2) .≈ 1)
    @test all(y["EOC_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["PICS_off"]["mean"] .<= 1)
    @test all(y["PICS_off"]["mean"][y["EOC_off"]["mean"] .== 0.0] .== 0)
    @test all(sum(y["Wfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["XEOC_on"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_on"]["mean"] .<= 1)
    @test all(y["XPICS_on"]["mean"][y["XEOC_on"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["Xfrac_on"]["mean"] .<= 1)
    @test all(0 .<= y["XWfrac_on"]["mean"] .<= 1)
    @test all(y["XEOC_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_off"]["mean"] .<= 1)
    @test all(y["XPICS_off"]["mean"][y["XEOC_off"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["XWfrac_off"]["mean"] .<= 1)
    @test all(sum(y["XWfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["labelsfrac"]["mean"] .== labels)
    @test all(y["Nactivelabels"]["mean"] .== sum(labels))
end

@testset "simulation_stochastic_parallel" begin
    addprocs(2)
    @everywhere using Pkg
    @everywhere Pkg.activate(".")
    @everywhere using ContextualBandits
    FX = CovariatesCopula([Categorical([1/2,1/2]),OrdinalDiscrete([1/3,1/3,1/3])],0.3)
    FXtilde = FX
    Wn = 3
    m = length(FX)
    T = 10
    delay = 0
    labels = ones(Bool, (Wn+1)*m)
    theta0 = zeros(sum(labels))
    Sigma0 = Diagonal(ones(sum(labels)))
    sample_std = 1.0
    policy = RandomPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)
    policy2 = GreedyPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)
    policies = Dict("random" => policy, "greedy" => policy2)
    outcome_model = LinearOutcomeLabelRandom(Wn, m, labels, theta0, Sigma0, sample_std)
    reps = 100
    post_reps = 10
    pilot_samples_per_treatment = 0
    Xinterest = rand(FXtilde,2)
    rng = MersenneTwister(1234)
    x = simulation_stochastic_parallel(FX,FXtilde,Wn,T,delay,policies,outcome_model;
        reps=reps,post_reps=post_reps, pilot_samples_per_treatment = pilot_samples_per_treatment,Xinterest=Xinterest,rng=rng,verbose=true)
    y = x["output"]["greedy"]
    @test all(y["EOC_on"]["mean"] .>= 0.0)
    @test all(cumsum(y["EOC_on"]["mean"]) .≈ y["cumulEOC_on"]["mean"])
    @test all(0 .<= y["PICS_on"]["mean"] .<= 1)
    @test all(y["PICS_on"]["mean"][y["EOC_on"]["mean"] .== 0.0] .== 0)
    @test all(cumsum(y["PICS_on"]["mean"]) .≈ y["cumulPICS_on"]["mean"])
    @test all(sum(y["Wfrac_on"]["mean"],dims=2) .≈ 1)
    @test all(y["EOC_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["PICS_off"]["mean"] .<= 1)
    @test all(y["PICS_off"]["mean"][y["EOC_off"]["mean"] .== 0.0] .== 0)
    @test all(sum(y["Wfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["XEOC_on"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_on"]["mean"] .<= 1)
    @test all(y["XPICS_on"]["mean"][y["XEOC_on"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["Xfrac_on"]["mean"] .<= 1)
    @test all(0 .<= y["XWfrac_on"]["mean"] .<= 1)
    @test all(y["XEOC_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_off"]["mean"] .<= 1)
    @test all(y["XPICS_off"]["mean"][y["XEOC_off"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["XWfrac_off"]["mean"] .<= 1)
    @test all(sum(y["XWfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["labelsfrac"]["mean"] .== labels)
    @test all(y["Nactivelabels"]["mean"] .== sum(labels))
end