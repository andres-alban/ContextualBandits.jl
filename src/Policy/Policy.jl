"""
    abstract type Policy

Supertype for contextual bandit policies.

The functions [`initialize!`](@ref), [`state_update!`](@ref), [`allocation`](@ref), and [`implementation`](@ref)
take subtypes of `Policy` as the first argument. Each subtype of Policy should implement these functions.
"""
abstract type Policy end

"""
    initialize!(policy::Policy,W,X,Y)

Initialize the state of a policy before a trial starts. `W`, `X`, and `Y` is data
collected in a pilot that can be used to initialize the policy. 
`W` is the vector of treatments, `X` is the matrix of covariates, and `Y` is the vector of outcomes.
"""
function initialize!(policy::Policy,W,X,Y)
end

"""
    state_update!(policy::Policy,W,X,Y)

Update the state of a policy givent the data `W`, `X`, and `Y`. `W` is the vector of treatments, `X` is the matrix of covariates, and `Y` is the vector of outcomes.

For example, the policy may do Bayesian updating to get posterior parameters.
"""
function state_update!(policy::Policy,W,X,Y)
end

"""
    allocation(policy::Policy,Xcurrent,W,X,Y,rng=Random.GLOBAL_RNG)

Return a treatment to allocate a patient with covariates `Xcurrent`, given that the trial has observed `W`, `X`, and `Y`.
`W` is the vector of treatments, `X` is the matrix of covariates, and `Y` is the vector of outcomes.

The dimension of `Y` may be smaller than that of `W` and `X` because of delays in outcomes.
"""
function allocation(policy::Policy,Xcurrent,W,X,Y,rng=Random.GLOBAL_RNG)
    return 1
end

"""
    implementation(policy::Policy,X_post,W,X,Y)

Implement a treatment for covariates X_post given that the trial observed `W`, `X`, and `Y`.
`W` is the vector of treatments, `X` is the matrix of covariates, and `Y` is the vector of outcomes.
"""
function implementation(policy::Policy,X_post,W,X,Y)
    return ones(Int,size(X_post,2))
end