"""
    PolicyLinearDiscrete <: Policy

Abstract subtype of [PolicyLinear](@ref) that only accepts models that are discrete.
Covariate vectors are converted to group indices using [X2g](@ref).

All subtypes must include a `model::BayesLinearRegressionDiscrete` field.
"""
abstract type PolicyLinearDiscrete <: PolicyLinear end

function initialize!(policy::PolicyLinearDiscrete, W=Int[], X=Float64[], Y=Float64[])
    initialize!(policy.model, W, X, Y)
end

function state_update!(policy::PolicyLinearDiscrete, W, X, Y, rng=Random.default_rng())
    t = length(Y)
    if t > 0
        state_update!(policy.model, W[t], view(X,:,t), Y[t])
    end
end

function implementation(policy::PolicyLinearDiscrete, X_post, W, X, Y)
    P = size(X_post,2)
    treat_post = Vector{Int}(undef,P)
    for k in 1:P
        g = X2g(view(X_post,:,k), policy.model.FX)
        treat_post[k] = argmax(policy.model.theta_t[treatment_g2index.(1:policy.model.n, g, policy.model.gn)])
    end
    return treat_post
end