"""
    PolicyLinearDiscrete <: PolicyLinear

Abstract subtype of [PolicyLinear](@ref) that only accepts models that are discrete.
Covariate vectors are converted to group indices using [X2g](@ref).

All subtypes must include a `model::BayesLinearRegressionDiscrete` field.
"""
abstract type PolicyLinearDiscrete <: PolicyLinear end

function initialize!(policy::PolicyLinearDiscrete, W=Int[], X=Float64[], Y=Float64[])
    initialize!(policy.model, W, X, Y)
end

function state_update!(policy::PolicyLinearDiscrete, W, X, Y)
    state_update!(policy.model, W, X, Y)
end

function implementation(policy::PolicyLinearDiscrete, X_post, W, X, Y)
    n = size(X_post,2)
    treat_post = Vector{Int}(undef,n)
    for k in 1:n
        g = X2g(view(X_post,:,k), policy.model.FX)
        treat_post[k] = argmax(policy.model.theta_t[treatment_g2index.(1:policy.model.Wn, g, policy.model.gn)])
    end
    return treat_post
end