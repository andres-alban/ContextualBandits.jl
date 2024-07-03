"""
    PolicyLinear <: Policy

Abstract supertype that updates using the model with a labeling and implements the treatments strategy that maximizes expected outcomes.
It does not provide an allocation policy, which should be defined for each subtype.

All subtypes must include a `model::BayesLinearRegression` field.
"""
abstract type PolicyLinear <: Policy end

function initialize!(policy::PolicyLinear,W=Int[],X=Float64[],Y=Float64[])
    initialize!(policy.model,W,X,Y)
end

function state_update!(policy::PolicyLinear,W,X,Y,rng=Random.GLOBAL_RNG)
    t = length(Y)
    if t > 0
        state_update!(policy.model,W[t],view(X,:,t),Y[t])
    end
end

function implementation(policy::PolicyLinear,X_post,W,X,Y)
    n = size(X_post,2)
    treat_post = Vector{Int}(undef,n)
    for k in 1:n 
        treat_post[k] = argmax([interact(iw, policy.model.Wn, view(X_post,:,k), policy.model.labeling)' * policy.model.theta_t for iw in 1:policy.model.Wn])
    end
    return treat_post
end

function policy_labeling(policy::PolicyLinear)
    model_labeling(policy.model)
end

"""
    RandomPolicyLinear <: PolicyLinear
    RandomPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m),trues(Wn*m)))

Allocate treatment uniformly at random.
Use a linear model (with `labeling`) to make an implementation.
"""
struct RandomPolicyLinear <: PolicyLinear
    model::BayesLinearRegression
end

function RandomPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m),trues(Wn*m)))
    RandomPolicyLinear(BayesLinearRegression(Wn, m, theta0, Sigma0, sample_std, labeling))
end

function allocation(policy::RandomPolicyLinear,Xcurrent,W,X,Y,rng=Random.GLOBAL_RNG)
    rand(rng,1:policy.model.Wn)
end

"""
    GreedyPolicyLinear <: PolicyLinear
    GreedyPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m),trues(Wn*m)))

Allocate and implement the treatment with the largest expected outcome based on 
a linear model (with `labeling`).
"""
struct GreedyPolicyLinear <: PolicyLinear
    model::BayesLinearRegression
end

function GreedyPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m),trues(Wn*m)))
    GreedyPolicyLinear(BayesLinearRegression(Wn, m, theta0, Sigma0, sample_std, labeling))
end

function allocation(policy::GreedyPolicyLinear,Xcurrent,W,X,Y,rng=Random.GLOBAL_RNG)
    return argmax_ties([interact(iw, policy.model.Wn, Xcurrent, policy.model.labeling)' * policy.model.theta_t for iw in 1:policy.model.Wn], rng)
end