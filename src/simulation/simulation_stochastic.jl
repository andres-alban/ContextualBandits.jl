"""
    simulation_stochastic_internal(FX, FXtilde, Wn, T, delay, policies, outcome_model;
        reps=100, post_reps=100, recorder=StandardRecorder(), aggregators=nothing, pilot_samples_per_treatment = 0,
        Xinterest=zeros(length(FX),0), rng=Random.GLOBAL_RNG, verbose=false)

Simulate trials and record metrics. This function is not intended to be called directly,
but rather through [simulation_stochastic](@ref) and [simulation_stochastic_parallel](@ref).
"""
function simulation_stochastic_internal(FX, FXtilde, Wn, T, delay, policies, outcome_model;
    reps=100, post_reps=100, recorder=StandardRecorder(), aggregators=nothing, pilot_samples_per_treatment = 0,
    Xinterest=zeros(length(FX),0), rng=Random.GLOBAL_RNG, verbose=false)

    @assert length(FX) == length(FXtilde)

    X = Matrix{Float64}(undef,length(FX),T)
    finite, weights, X_post = try
        finite_groups(FXtilde,post_reps)
    catch
        false, ones(post_reps)./post_reps, Matrix{Float64}(undef,length(FXtilde),post_reps)
    end

    initialize!(recorder, T, Wn, length(FX), delay, size(X_post,2), weights, size(Xinterest,2))
    if isnothing(aggregators)
        aggregators = [StandardResultsAggregator(output_recorder(recorder),output_recorder_names(recorder)) for _ in policies]
    end

    Wpilot = repeat(1:Wn,pilot_samples_per_treatment)
    Xpilot = Matrix{Float64}(undef, length(FX), pilot_samples_per_treatment*Wn)
    Ypilot = Vector{Float64}(undef, pilot_samples_per_treatment*Wn)

    for i in 1:reps
        rand!(rng,FX,X)
        if !finite
            rand!(rng,FXtilde,X_post)
        end
        outcome_model_state!(outcome_model,rng)
        Z = [noise_outcome(outcome_model,rng) for t in 1:T]
        if pilot_samples_per_treatment > 0
            rand!(rng,FX,Xpilot)
            Ypilot .= [noisy_outcome(outcome_model, Wpilot[i], view(Xpilot, :, i), noise_outcome(outcome_model,rng)) for i in eachindex(Wpilot)]
        end
        rng_policy = MersenneTwister(abs(rand(rng,Int)))
        for (ip,policy) in enumerate(policies)
            initialize!(policy,Wpilot,Xpilot,Ypilot)
            reset!(recorder,outcome_model,X_post,Xinterest)
            out = replication_stochastic(X,X_post,Z,Wn,delay,policy,outcome_model,recorder;Xinterest=Xinterest,X_post_weights=weights,rng=copy(rng_policy))
            update!(aggregators[ip], out)
        end
        if verbose && (i % 100 == 0)
            println(i)
        end
    end

    return aggregators
end

"""
    finite_groups(FXtilde,post_reps)

For a Covariates generator `FXtilde` (most likely an instance of [CovariatesIndependent](@ref) or [CovariatesCopula](@ref)),
check if all covariates are finite and returns a flag indicating if the covariates are finite, the probability weights of each covariates value,
and a matrix of the covariates. If the covariates are not finite or the number of possible covariates is larger than `post_reps`, 
the weights are all equal to `1/post_reps`, and the matrix is of the right dimensions but uninitialized. 
"""
function finite_groups(FXtilde,post_reps)
    for i in 1:length(marginals(FXtilde))
        if !(typeof(marginals(FXtilde)[i]) <: Categorical) && !(typeof(marginals(FXtilde)[i]) <: OrdinalDiscrete)
            return false, ones(post_reps)./post_reps, Matrix{Float64}(undef,length(FXtilde),post_reps)
        end
    end
    weights = X2g_probs(FXtilde)
    if length(weights) > post_reps # If ptilde is too large (there are too many groups), we fall back to random draws
        return false, ones(post_reps)./post_reps, Matrix{Float64}(undef,length(FXtilde),post_reps)
    end

    X_post = Matrix{Float64}(undef,length(FXtilde),length(weights))
    for g in eachindex(weights)
        X_post[:,g] = g2X(g,FXtilde)
    end
    return true, weights, X_post
end

"""
    simulation_stochastic(FX, FXtilde, Wn, T, delay, policies, outcome_model;
        reps=100, post_reps=100, recorder=StandardRecorder(), aggregators=nothing, pilot_samples_per_treatment = 0, Xinterest=zeros(length(FX),0), rng=Random.GLOBAL_RNG, verbose=false)

Simulate trials and record metrics.

# Arguments
- `FX`: Covariates generator for in-trial covariates (e.g., [CovariatesIndependent](@ref) or [CovariatesCopula](@ref)).
- `FXtilde`: Covariates generator for post-trial covariates (e.g., [CovariatesIndependent](@ref) or [CovariatesCopula](@ref)).
- `Wn`: Number of treatments.
- `T`: Number of patients in the trial (sample size).
- `delay`: Delay in observing outcomes.
- `policies`: Dictionary of policies to simulate. The keys are the names of the policies and the values are the policies themselves.
- `outcome_model`: the model that generates the outcomes (e.g., [LinearOutcomeLabelRandom](@ref)).
- `reps`: Number of replications.
- `post_reps`: Number of replications for post-trial covariates.
- `recorder`: Recorder to store the metrics. Default is [StandardRecorder](@ref). Advanced users can create their own recorders.
- `aggregators`: Aggregators to summarize the metrics from recorder. Default is `nothing`, 
which means that a [StandardResultsAggregator](@ref) will be created for each policy. StandardResultsAggregator stores the mean and standard deviation of the metrics.
- `pilot_samples_per_treatment`: Number of pilot samples per treatment to build a prior distribution before the start of the trial. Default is 0.
- `Xinterest`: Covariates of interest for which specific . Default is an empty matrix.
- `rng`: Random number generator. Default is `Random.GLOBAL_RNG`.
- `verbose`: Print progress of the simulation. Default is `false`.

# Returns
A dictionary with the input arguments and the output of the simulation for each policy. The output of the simulation depends on the recorder and aggregator.
"""
function simulation_stochastic(FX, FXtilde, Wn, T, delay, policies, outcome_model;
    reps=100, post_reps=100, recorder=StandardRecorder(), aggregators=nothing, pilot_samples_per_treatment = 0,
    Xinterest=zeros(length(FX),0), rng=Random.GLOBAL_RNG, verbose=false)

    policy_labels = try
        [string(j) for j in keys(policies)]
    catch
        ["policy" * string(i) for i in 1:length(policies)]
    end
    policy_values = collect(values(policies))

    aggregators = simulation_stochastic_internal(FX, FXtilde, Wn, T, delay, policy_values, outcome_model;
    reps=reps, post_reps=post_reps, recorder=recorder, aggregators=aggregators, pilot_samples_per_treatment = pilot_samples_per_treatment,
    Xinterest=Xinterest, rng=rng, verbose=verbose)

    output = Dict(policy_labels[i] => asdict(aggregators[i]) for i in eachindex(policy_labels))

    return Dict(
        "input" => Dict("reps" => reps, "T" => T, "delay" => delay, "Xinterest" => Xinterest,
            "FX" => FX, "FXtilde" => FXtilde, "Wn" => Wn, "policies" => policies, "outcome_model" => outcome_model,
            "pilot_samples_per_treatment" => pilot_samples_per_treatment),
        "output" => output
    )
end

"""
    simulation_stochastic_parallel(FX, FXtilde, Wn, T, delay, policies, outcome_model;
        reps=100, post_reps=100, recorder=StandardRecorder(), aggregators=nothing, pilot_samples_per_treatment = 0, Xinterest=zeros(length(FX),0), rng=Random.GLOBAL_RNG, verbose=false)

Parallel version of [simulation_stochastic](@ref). The simulation is distributed among all available workers.

# Example
```julia
using Distributed
addprocs(2)
@everywhere using ContextualBandits
# ...
# generate all the input arguments for the function
# ...
results = simulation_stochastic_parallel(FX, FXtilde, Wn, T, delay, policies, outcome_model)
```
"""
function simulation_stochastic_parallel(FX, FXtilde, Wn, T, delay, policies, outcome_model;
    reps=100, post_reps=100, recorder=StandardRecorder(), aggregators=nothing, pilot_samples_per_treatment = 0,
    Xinterest=zeros(length(FX),0), rng=Random.GLOBAL_RNG, verbose=false)

    policy_labels = try
        [string(j) for j in keys(policies)]
    catch
        ["policy" * string(i) for i in 1:length(policies)]
    end
    policy_values = collect(values(policies))

    wrkrs = workers()
    nwrkrs = length(wrkrs)
    futures = Vector{Future}(undef,nwrkrs)
    reps_per_worker = div(reps,nwrkrs)
    rem_reps = reps % nwrkrs # remaining reps will be distributed among the first workers that receive the task
    for (i,w) in enumerate(wrkrs)
        rng = randjump(rng, big(10)^20)
        repsworker = reps_per_worker + (rem_reps > 0)
        futures[i] = @spawnat w simulation_stochastic_internal(FX, FXtilde, Wn, T, delay, policy_values, outcome_model;
            reps=repsworker, post_reps=post_reps, recorder=recorder, aggregators=aggregators, pilot_samples_per_treatment = pilot_samples_per_treatment,
            Xinterest=Xinterest, rng=rng, verbose=verbose)
        rem_reps -= 1
    end

    results = Vector{Vector{<:SimulationResultsAggregator}}(undef,nwrkrs)
    for i in 1:nwrkrs
        results[i] = fetch(futures[i])
    end

    output = results[1]
    for i in 2:nwrkrs
        for j in eachindex(output)
            update!(output[j],results[i][j])
        end
    end

    output = Dict(policy_labels[i] => asdict(output[i]) for i in eachindex(policy_labels))

    return Dict(
        "input" => Dict("reps" => reps, "T" => T, "delay" => delay, "Xinterest" => Xinterest,
            "FX" => FX, "FXtilde" => FXtilde, "Wn" => Wn, "policies" => policies, "outcome_model" => outcome_model,
            "pilot_samples_per_treatment" => pilot_samples_per_treatment),
        "output" => output
    )
end