"""
    struct RandEpsilon{T<:Policy} <: Policy
    RandEpsilon(subpolicy, n, epsilon)

Allocate following `policy` with probability `1-epsilon` or sample at random among all alternatives otherwise.
"""
struct RandEpsilon{T<:Policy} <: Policy
    subpolicy::T
    n::Int
    epsilon::Float64
end

function initialize!(policy::RandEpsilon, W, X, Y)
    initialize!(policy.subpolicy, W, X, Y)
end

function state_update!(policy::RandEpsilon, W, X, Y, rng=Random.default_rng())
    state_update!(policy.subpolicy, W, X, Y)
end

function allocation(policy::RandEpsilon, Xcurrent, W, X, Y, rng=Random.GLOBAL_RNG)
    if rand(rng) < policy.epsilon
        return rand(rng, 1:policy.n)
    else
        return allocation(policy.subpolicy, Xcurrent, W, X, Y, rng)
    end
end

function implementation(policy::RandEpsilon, X_post, W, X, Y)
    return implementation(policy.subpolicy, X_post, W, X, Y)
end