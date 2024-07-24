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
    n = 5
    X = rand(m, T)
    X_post = rand(m, 4)
    Z = rand(T)
    delay = 0
    labeling = ones(Bool, (n+1)*m)
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    policy = RandomPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    ContextualBandits.initialize!(policy, [],[],[])
    outcome_model = OutcomeLinearBayes(n, m, theta0, Sigma0, sample_std, labeling)
    ContextualBandits.outcome_model_state!(outcome_model)
    Xinterest = X[:,3:4]
    X_post_weights = ones(4)./4
    recorder = ContextualBandits.StandardRecorder()
    ContextualBandits.initialize!(recorder, T, n, size(X,1), delay, size(X_post,2), X_post_weights, size(Xinterest,2))
    ContextualBandits.reset!(recorder,outcome_model,X_post,Xinterest)
    rng = MersenneTwister(1234)
    x = ContextualBandits.replication_stochastic(X,X_post,Z,n,delay,policy,outcome_model,recorder;Xinterest=Xinterest,X_post_weights=X_post_weights,rng=rng)
    names = ContextualBandits.output_recorder_names(recorder)
    @test all(x[findfirst(names .== "regret_on")] .>= 0.0)
    @test all(cumsum(x[findfirst(names .== "regret_on")]) .≈ x[findfirst(names .== "cumulregret_on")])
    @test all(x[findfirst(names .== "PICS_on")] .∈ Ref([0,1]))
    @test all(x[findfirst(names .== "PICS_on")][x[findfirst(names .== "regret_on")] .== 0.0] .== 0)
    @test all(cumsum(x[findfirst(names .== "PICS_on")]) .== x[findfirst(names .== "cumulPICS_on")])
    @test all(sum(x[findfirst(names .== "Wfrac_on")],dims=2) .== 1)
    @test all(x[findfirst(names .== "regret_off")] .>= 0.0)
    @test all(0 .<= x[findfirst(names .== "PICS_off")] .<= 1)
    @test all(x[findfirst(names .== "PICS_off")][x[findfirst(names .== "regret_off")] .== 0.0] .== 0)
    @test all(sum(x[findfirst(names .== "Wfrac_off")],dims=2) .== 1)
    @test all((x[findfirst(names .== "Xregret_on")] .>= 0.0) .| isnan.(x[findfirst(names .== "Xregret_on")]))
    @test all((x[findfirst(names .== "XPICS_on")] .∈ Ref([0,1])) .| isnan.(x[findfirst(names .== "XPICS_on")]))
    @test all(x[findfirst(names .== "XPICS_on")][x[findfirst(names .== "Xregret_on")] .== 0.0] .== 0)
    @test all(x[findfirst(names .== "Xfrac_on")] .∈ Ref([0,1]))
    @test all(x[findfirst(names .== "XWfrac_on")] .∈ Ref([0,1]))
    @test all(x[findfirst(names .== "Xregret_off")] .>= 0.0)
    @test all(x[findfirst(names .== "XPICS_off")] .∈ Ref([0,1]))
    @test all(x[findfirst(names .== "XPICS_off")][x[findfirst(names .== "Xregret_off")] .== 0.0] .== 0)
    @test all(x[findfirst(names .== "XWfrac_off")] .∈ Ref([0,1]))
    @test all(sum(x[findfirst(names .== "XWfrac_off")],dims=2) .== 1)
    @test all(x[findfirst(names .== "labeling_frac")] .== labeling)
    @test all(x[findfirst(names .== "sum_labeling")] .== sum(labeling))
end

@testset "simulation_stochastic_internal" begin
    FX = CovariatesIndependent([Normal(),Normal()])
    FXtilde = FX
    n = 3
    m = length(FX)
    T = 10
    delay = 0
    labeling = ones(Bool, (n+1)*m)
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    policy = RandomPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    policy2 = GreedyPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    policies = [policy, policy2]
    outcome_model = OutcomeLinearBayes(n, m, theta0, Sigma0, sample_std, labeling)
    reps = 10
    post_reps = 10
    pilot_samples_per_treatment = 0
    Xinterest = rand(FXtilde,2)
    recorder = ContextualBandits.StandardRecorder()
    weights = ones(post_reps)./post_reps
    ContextualBandits.initialize!(recorder, T, n, length(FX), delay, post_reps, weights, size(Xinterest,2))
    aggregators = aggregators = [ContextualBandits.StandardResultsAggregator(ContextualBandits.output_recorder(recorder),ContextualBandits.output_recorder_names(recorder)) for _ in policies]
    rng = MersenneTwister(1234)
    x = ContextualBandits.simulation_stochastic_internal(FX,FXtilde,n,T,delay,policies,outcome_model,
        reps, post_reps, recorder, aggregators, pilot_samples_per_treatment, Xinterest, rng, false)
    y = ContextualBandits.asdict(x[2])
    @test all(y["regret_on"]["mean"] .>= 0.0)
    @test all(cumsum(y["regret_on"]["mean"]) .≈ y["cumulregret_on"]["mean"])
    @test all(0 .<= y["PICS_on"]["mean"] .<= 1)
    @test all(y["PICS_on"]["mean"][y["regret_on"]["mean"] .== 0.0] .== 0)
    @test all(cumsum(y["PICS_on"]["mean"]) .≈ y["cumulPICS_on"]["mean"])
    @test all(sum(y["Wfrac_on"]["mean"],dims=2) .≈ 1)
    @test all(y["regret_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["PICS_off"]["mean"] .<= 1)
    @test all(y["PICS_off"]["mean"][y["regret_off"]["mean"] .== 0.0] .== 0)
    @test all(sum(y["Wfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["Xregret_on"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_on"]["mean"] .<= 1)
    @test all(y["XPICS_on"]["mean"][y["Xregret_on"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["Xfrac_on"]["mean"] .<= 1)
    @test all(0 .<= y["XWfrac_on"]["mean"] .<= 1)
    @test all(y["Xregret_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_off"]["mean"] .<= 1)
    @test all(y["XPICS_off"]["mean"][y["Xregret_off"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["XWfrac_off"]["mean"] .<= 1)
    @test all(sum(y["XWfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["labeling_frac"]["mean"] .== labeling)
    @test all(y["sum_labeling"]["mean"] .== sum(labeling))
end

@testset "simulation_stochastic" begin
    FX = CovariatesCopula([Categorical([1/2,1/2]),OrdinalDiscrete([1/3,1/3,1/3])],0.3)
    FXtilde = FX
    n = 3
    m = length(FX)
    T = 10
    delay = 0
    labeling = ones(Bool, (n+1)*m)
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    policy = RandomPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    policy2 = GreedyPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    policies = Dict("random" => policy, "greedy" => policy2)
    outcome_model = OutcomeLinearBayes(n, m, theta0, Sigma0, sample_std, labeling)
    reps = 10
    post_reps = 10
    pilot_samples_per_treatment = 0
    Xinterest = rand(FXtilde,2)
    rng = MersenneTwister(1234)
    x = simulation_stochastic(reps, FX,n,T,policies,outcome_model; FXtilde, delay,
        post_reps=post_reps, pilot_samples_per_treatment = pilot_samples_per_treatment,Xinterest=Xinterest,rng=rng,verbose=true)
    y = x["output"]["random"]
    @test all(y["regret_on"]["mean"] .>= 0.0)
    @test all(cumsum(y["regret_on"]["mean"]) .≈ y["cumulregret_on"]["mean"])
    @test all(0 .<= y["PICS_on"]["mean"] .<= 1)
    @test all(y["PICS_on"]["mean"][y["regret_on"]["mean"] .== 0.0] .== 0)
    @test all(cumsum(y["PICS_on"]["mean"]) .≈ y["cumulPICS_on"]["mean"])
    @test all(sum(y["Wfrac_on"]["mean"],dims=2) .≈ 1)
    @test all(y["regret_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["PICS_off"]["mean"] .<= 1)
    @test all(y["PICS_off"]["mean"][y["regret_off"]["mean"] .== 0.0] .== 0)
    @test all(sum(y["Wfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["Xregret_on"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_on"]["mean"] .<= 1)
    @test all(y["XPICS_on"]["mean"][y["Xregret_on"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["Xfrac_on"]["mean"] .<= 1)
    @test all(0 .<= y["XWfrac_on"]["mean"] .<= 1)
    @test all(y["Xregret_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_off"]["mean"] .<= 1)
    @test all(y["XPICS_off"]["mean"][y["Xregret_off"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["XWfrac_off"]["mean"] .<= 1)
    @test all(sum(y["XWfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["labeling_frac"]["mean"] .== labeling)
    @test all(y["sum_labeling"]["mean"] .== sum(labeling))
end

@testset "simulation_stochastic_parallel" begin
    addprocs(2)
    @everywhere using Pkg
    @everywhere Pkg.activate(".")
    @everywhere using ContextualBandits
    FX = CovariatesCopula([Categorical([1/2,1/2]),OrdinalDiscrete([1/3,1/3,1/3])],0.3)
    FXtilde = FX
    n = 3
    m = length(FX)
    T = 10
    delay = 0
    labeling = ones(Bool, (n+1)*m)
    theta0 = zeros(sum(labeling))
    Sigma0 = Diagonal(ones(sum(labeling)))
    sample_std = 1.0
    policy = RandomPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    policy2 = GreedyPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    policies = Dict("random" => policy, "greedy" => policy2)
    outcome_model = OutcomeLinearBayes(n, m, theta0, Sigma0, sample_std, labeling)
    reps = 100
    post_reps = 10
    pilot_samples_per_treatment = 0
    Xinterest = rand(FXtilde,2)
    rng = MersenneTwister(1234)
    x = simulation_stochastic_parallel(reps,FX,n,T,policies,outcome_model; FXtilde, delay,
        post_reps=post_reps, pilot_samples_per_treatment = pilot_samples_per_treatment,Xinterest=Xinterest,rng=rng,verbose=true)
    y = x["output"]["greedy"]
    @test all(y["regret_on"]["mean"] .>= 0.0)
    @test all(cumsum(y["regret_on"]["mean"]) .≈ y["cumulregret_on"]["mean"])
    @test all(0 .<= y["PICS_on"]["mean"] .<= 1)
    @test all(y["PICS_on"]["mean"][y["regret_on"]["mean"] .== 0.0] .== 0)
    @test all(cumsum(y["PICS_on"]["mean"]) .≈ y["cumulPICS_on"]["mean"])
    @test all(sum(y["Wfrac_on"]["mean"],dims=2) .≈ 1)
    @test all(y["regret_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["PICS_off"]["mean"] .<= 1)
    @test all(y["PICS_off"]["mean"][y["regret_off"]["mean"] .== 0.0] .== 0)
    @test all(sum(y["Wfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["Xregret_on"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_on"]["mean"] .<= 1)
    @test all(y["XPICS_on"]["mean"][y["Xregret_on"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["Xfrac_on"]["mean"] .<= 1)
    @test all(0 .<= y["XWfrac_on"]["mean"] .<= 1)
    @test all(y["Xregret_off"]["mean"] .>= 0.0)
    @test all(0 .<= y["XPICS_off"]["mean"] .<= 1)
    @test all(y["XPICS_off"]["mean"][y["Xregret_off"]["mean"] .== 0.0] .== 0)
    @test all(0 .<= y["XWfrac_off"]["mean"] .<= 1)
    @test all(sum(y["XWfrac_off"]["mean"],dims=2) .≈ 1)
    @test all(y["labeling_frac"]["mean"] .== labeling)
    @test all(y["sum_labeling"]["mean"] .== sum(labeling))
end