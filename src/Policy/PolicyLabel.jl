"""
    abstract type PolicyLabel <: Policy

An abstract type that updates using the model with a labeling and implements the treatments strategy that maximizes expected outcomes.
It does not provide an allocation policy, which should be defined for each subtype.

All subtypes must include the following fields: 
- `theta0` (prior mean vector)
- `Sigma0` (prior covariance matrix)
- `theta_t` (posterior mean vector)
- `Sigma_t` (posterior covariance matrix)
- `labels` (labeling)
"""
abstract type PolicyLabel <: Policy end

function initialize!(policy::PolicyLabel,W,X,Y)
    if length(Y) == 0
        policy.theta_t = copy(policy.theta0)
        policy.Sigma_t = copy(policy.Sigma0)
    else # pilot data to build the prior
        WX = interact(W, policy.Wn, X, policy.labels)
        policy.theta_t, policy.Sigma_t = BayesUpdateNormal(policy.theta0, policy.Sigma0, WX, Y, policy.sample_std)
        robustify_prior!(policy.theta_t, policy.Sigma_t, policy.Wn, policy.labels)
    end
    return
end

function state_update!(policy::PolicyLabel,W,X,Y)
    if !isempty(Y)
        index = length(Y)
        BayesUpdateNormal!(policy.theta_t,policy.Sigma_t,interact(W[index],policy.Wn,X[:,index],policy.labels),Y[index],policy.sample_std)
    end
    return
end

function implementation(policy::PolicyLabel,X_post,W,X,Y)
    n = size(X_post,2)
    treat_post = Vector{Int}(undef,n)
    for k in 1:n 
        treat_post[k] = argmax_ties([interact(iw,policy.Wn,view(X_post,:,k), policy.labels)' * policy.theta_t for iw in 1:policy.Wn])
    end
    return treat_post
end

function policy_labeling(policy::PolicyLabel)
    return policy.labels
end


"""
    mutable struct RandomPolicyLabel <: PolicyLabel
    RandomPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)

Allocate treatment uniformly at random and update based on model with labeling `labels` to make an implementation.
"""
mutable struct RandomPolicyLabel <: PolicyLabel
    Wn::Int
    labels::BitArray{1}
    sample_std::Float64
    theta0::Vector{Float64}
    Sigma0::Array{Float64,2}
    theta_t::Vector{Float64}
    Sigma_t::Array{Float64,2}
    function RandomPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)
        d = sum(labels)
        length(theta0) == d || throw(DomainError(theta0,"`theta0` must be of length `sum(labels)`."))
        size(Sigma0) == (d,d) || throw(DomainError(Sigma0,"`Sigma0` must be of dimensions `sum(labels)`."))
        issymmetric(Sigma0) || throw(DomainError(Sigma0,"`Sigma0` must be symmetric."))
        mineigval = minimum(eigvals(Sigma0))
        mineigval >= -sqrt(eps(Float64)) || throw(DomainError(Sigma0,"`Sigma0` must be positive semidefinite."))
        mineigval > 0 || @warn "`Sigma0` is semi-definite. Numerical errors are possible."
        new(Wn, copy(labels), sample_std, copy(theta0), copy(Sigma0), copy(theta0), copy(Sigma0))
    end
end

function allocation(policy::RandomPolicyLabel,Xcurrent,W,X,Y,rng=Random.GLOBAL_RNG)
    rand(rng,1:policy.Wn)
end

"""
    mutable struct GreedyPolicyLabel <: PolicyLabel
    GreedyPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)

Allocate and implement the treatment with the largest expected outcome based on predprog model with labels.
"""
mutable struct GreedyPolicyLabel <: PolicyLabel
    Wn::Int
    labels::BitArray{1}
    sample_std::Float64
    theta0::Vector{Float64}
    Sigma0::Array{Float64,2}
    theta_t::Vector{Float64}
    Sigma_t::Array{Float64,2}
    function GreedyPolicyLabel(Wn, labels, sample_std, theta0, Sigma0)
        d = sum(labels)
        length(theta0) == d || throw(DomainError(theta0,"`theta0` must be of length `sum(labels)`."))
        size(Sigma0) == (d,d) || throw(DomainError(Sigma0,"`Sigma0` must be of dimensions `sum(labels)`."))
        issymmetric(Sigma0) || throw(DomainError(Sigma0,"`Sigma0` must be symmetric."))
        mineigval = minimum(eigvals(Sigma0))
        mineigval >= -sqrt(eps(Float64)) || throw(DomainError(Sigma0,"`Sigma0` must be positive semidefinite."))
        mineigval > 0 || @warn "`Sigma0` is semi-definite. Numerical errors are possible."
        new(Wn, copy(labels), sample_std, copy(theta0), copy(Sigma0), copy(theta0), copy(Sigma0))
    end
end

function allocation(policy::GreedyPolicyLabel,Xcurrent,W,X,Y,rng=Random.GLOBAL_RNG)
    return argmax_ties([interact(iw,policy.Wn,Xcurrent,policy.labels)' * policy.theta_t for iw in 1:policy.Wn],rng)
end