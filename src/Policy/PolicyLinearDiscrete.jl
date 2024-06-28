"""
    PolicyLinearDiscrete <: PolicyLinear

Abstract subtype of [PolicyLinear](@ref) that only accepts models that are discrete.
Covariate vectors are converted to group indices using [X2g](@ref).

In addition to the required entries for [PolicyLinear](@ref), the following fields are required:
- FX (covariate distribution, e.g., [CovariatesIndependent](@ref) or [CovariatesCopula](@ref))
"""
abstract type PolicyLinearDiscrete <: PolicyLinear end

function initialize!(policy::PolicyLinearDiscrete,W,X,Y)
    if length(Y) > 0
        WX = interact(W, policy.Wn, X, policy.labeling)
        theta, Sigma = BayesUpdateNormal(policy.theta0, policy.Sigma0, WX, Y, policy.sample_std)
        robustify_prior!(theta, Sigma, policy.Wn, policy.labeling)
        policy.theta_t, policy.Sigma_t = X2g_prior(theta, Sigma, policy.FX, policy.labeling, policy.Wn)
    else
        policy.theta_t, policy.Sigma_t = X2g_prior(policy.theta0, policy.Sigma0, policy.FX, policy.labeling, policy.Wn)
    end
    return
end

function state_update!(policy::PolicyLinearDiscrete,W,X,Y)
    if isempty(Y)
        return
    end
    index = length(Y)
    g = X2g(view(X,:,index),policy.FX)
    BayesUpdateNormalDiscrete!(policy.theta_t, policy.Sigma_t, treatment_g2index(W[index], g, policy.gn), Y[index], policy.sample_std)
    return
end

function implementation(policy::PolicyLinearDiscrete,X_post,W,X,Y)
    n = size(X_post,2)
    treat_post = Vector{Int}(undef,n)
    for k in 1:n
        g = X2g(view(X_post,:,k),policy.FX)
        treat_post[k] = argmax(policy.theta_t[treatment_g2index.(1:policy.Wn, g, policy.gn)])
    end
    return treat_post
end

function checkInputPolicyLinearDiscrete(Wn, m, theta0, Sigma0, sample_std, labeling, FX)
    m == length(FX) || throw(DomainError("The number of covariates m must be the same as the length of FX."))
    checkInputPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling)
    for marginal in marginals(FX)
        typeof(marginal) <: Categorical || typeof(marginal) <: OrdinalDiscrete || throw(DomainError("marginal distributions need to be either `Categorical` or `OrdinalDiscrete`."))
    end
end
