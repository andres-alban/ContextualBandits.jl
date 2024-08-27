"""
    Policy

Supertype for contextual bandit policies.

The functions [`initialize!`](@ref), [`state_update!`](@ref), [`allocation`](@ref), and [`implementation`](@ref)
take subtypes of `Policy` as the first argument. Each subtype of Policy should implement these functions.
"""
abstract type Policy end

"""
    initialize!(policy::Policy[, W, X, Y])

Initialize the state of a policy before a trial starts. `W`, `X`, and `Y` is data
collected in a pilot that can be used to initialize the policy. 
`W` is the vector of treatments, `X` is the matrix of covariates, and `Y` is the vector of outcomes.
"""
function initialize!(policy::Policy, W=Int[], X=Float64[], Y=Float64[])
end

"""
    state_update!(policy::Policy,W,X,Y[, rng])

Update the state of a policy given the data `W`, `X`, and `Y`. `W` is the vector of treatments, `X` is the matrix of covariates, and `Y` is the vector of outcomes.

For example, the policy may do Bayesian updating to get posterior parameters.
"""
function state_update!(policy::Policy, W, X, Y, rng=Random.default_rng())
end

"""
    allocation(policy::Policy,Xcurrent,W,X,Y[, rng])

Return a treatment to allocate a patient with covariates `Xcurrent`, given that the trial has observed `W`, `X`, and `Y`.
`W` is the vector of treatments, `X` is the matrix of covariates, and `Y` is the vector of outcomes.

The dimension of `Y` may be smaller than that of `W` and `X` because of delays in outcomes.
"""
function allocation(policy::Policy, Xcurrent, W, X, Y, rng=Random.default_rng())
    return 1
end

"""
    implementation(policy::Policy,X_post,W,X,Y)

Implement a treatment for covariates X_post given that the trial observed `W`, `X`, and `Y`.
`W` is the vector of treatments, `X` is the matrix of covariates, and `Y` is the vector of outcomes.
"""
function implementation(policy::Policy, X_post, W, X, Y)
    return ones(Int, size(X_post, 2))
end

function allocationIndependent(policy::Policy, Xcurrent, W, X, Y, rng=Random.default_rng(), check=false, delay=0, Wpilot=[], Xpilot=[], Ypilot=[])
    initialize!(policy, Wpilot, Xpilot, Ypilot)
    T = length(Y)
    for t in 1:(T+delay)
        if check
            w = allocation(policy, Xcurrent, view(W, 1:(t-1)), view(X, :, 1:(t-1)), view(Y, 1:(t-delay-1)), rng)
            w == W[t] || @warn "The treatment in the data does not match the treatment allocated by the policy at time $t."
        end
        Wav = view(W, 1:min(t, T))
        Xav = view(X, :, 1:min(t, T))
        Yav = view(Y, 1:(t-delay))

        # Update state of policy
        state_update!(policy, Wav, Xav, Yav, rng)
    end
    return allocation(policy, Xcurrent, W, X, Y, rng)
end

function implementationIndependent(policy::Policy, X_post, W, X, Y, rng=Random.default_rng(), Wpilot=Int[], Xpilot=Float64[], Ypilot=Float64[])
    initialize!(policy, Wpilot, Xpilot, Ypilot)
    for t in eachindex(Y)
        state_update!(policy, view(W, 1:t), view(X, :, 1:t), view(Y, 1:t), rng)
    end
    return implementation(policy, X_post, W, X, Y)
end

# Some triavial policies

struct RandomPolicy <: Policy
    n::Int
end

function allocation(policy::RandomPolicy, Xcurrent, W, X, Y, rng=Random.default_rng())
    return rand(rng, 1:policy.n)
end


struct RoundRobinPolicy <: Policy
    n::Int
end

function allocation(policy::RoundRobinPolicy, Xcurrent, W, X, Y, rng=Random.default_rng())
    return (length(W) % policy.n) + 1
end