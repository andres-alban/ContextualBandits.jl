"""
    abstract type PolicyLinear <: Policy

An abstract type that updates using the model with a labeling and implements the treatments strategy that maximizes expected outcomes.
It does not provide an allocation policy, which should be defined for each subtype.

All subtypes must include the following fields:
- `Wn` (number of treatments)
- `theta0` (prior mean vector)
- `Sigma0` (prior covariance matrix)
- `theta_t` (posterior mean vector)
- `Sigma_t` (posterior covariance matrix)
- `labeling` (labeling)
"""
abstract type PolicyLinear <: Policy end

function initialize!(policy::PolicyLinear,W=[],X=[],Y=[])
    if length(Y) == 0
        policy.theta_t = copy(policy.theta0)
        policy.Sigma_t = copy(policy.Sigma0)
    else # pilot data to build the prior
        WX = interact(W, policy.Wn, X, policy.labeling)
        policy.theta_t, policy.Sigma_t = BayesUpdateNormal(policy.theta0, policy.Sigma0, WX, Y, policy.sample_std)
        robustify_prior!(policy.theta_t, policy.Sigma_t, policy.Wn, policy.labeling)
    end
    return
end

function state_update!(policy::PolicyLinear,W,X,Y)
    if !isempty(Y)
        index = length(Y)
        BayesUpdateNormal!(policy.theta_t,policy.Sigma_t,interact(W[index],policy.Wn,X[:,index],policy.labeling),Y[index],policy.sample_std)
    end
    return
end

function implementation(policy::PolicyLinear,X_post,W,X,Y)
    n = size(X_post,2)
    treat_post = Vector{Int}(undef,n)
    for k in 1:n 
        treat_post[k] = argmax_ties([interact(iw,policy.Wn,view(X_post,:,k), policy.labeling)' * policy.theta_t for iw in 1:policy.Wn])
    end
    return treat_post
end

function policy_labeling(policy::PolicyLinear)
    return policy.labeling
end

function checkInputPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling)
    d = sum(labeling)
    length(labeling) == (Wn+1)*m || throw(DomainError(labeling,"`labeling` must have length `(Wn+1)*m`."))
    length(theta0) == d || throw(DomainError(theta0,"`theta0` must be of length `sum(labeling)`."))
    size(Sigma0) == (d,d) || throw(DomainError(Sigma0,"`Sigma0` must be of dimensions `sum(labeling)`."))
    issymmetric(Sigma0) || throw(DomainError(Sigma0,"`Sigma0` must be symmetric."))
    mineigval = minimum(eigvals(Sigma0))
    mineigval >= -sqrt(eps(Float64)) || throw(DomainError(Sigma0,"`Sigma0` must be positive semidefinite."))
    mineigval > 0 || @warn "`Sigma0` is semi-definite. Numerical errors are possible."
    sample_std >= 0 || throw(DomainError(sample_std,"`sample_std` must be positive."))
    return true
end


"""
    mutable struct RandomPolicyLinear <: PolicyLinear
    RandomPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m),trues(Wn*m)))

Allocate treatment uniformly at random.
Use a linear model (with `labeling`) to make an implementation.
"""
mutable struct RandomPolicyLinear <: PolicyLinear
    Wn::Int
    theta0::Vector{Float64}
    Sigma0::Array{Float64,2}
    sample_std::Float64
    labeling::BitVector
    theta_t::Vector{Float64}
    Sigma_t::Matrix{Float64}
    function RandomPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m),trues(Wn*m)))
        checkInputPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling)
        new(Wn, copy(theta0), copy(Sigma0), sample_std, copy(labeling), similar(theta0), similar(Sigma0))
    end
end

function allocation(policy::RandomPolicyLinear,Xcurrent,W,X,Y,rng=Random.GLOBAL_RNG)
    rand(rng,1:policy.Wn)
end

"""
    mutable struct GreedyPolicyLinear <: PolicyLinear
    GreedyPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m),trues(Wn*m)))

Allocate and implement the treatment with the largest expected outcome based on 
a linear model (with `labeling`).
"""
mutable struct GreedyPolicyLinear <: PolicyLinear
    Wn::Int
    theta0::Vector{Float64}
    Sigma0::Matrix{Float64}
    sample_std::Float64
    labeling::BitVector
    theta_t::Vector{Float64}
    Sigma_t::Matrix{Float64}
    function GreedyPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m),trues(Wn*m)))
        checkInputPolicyLinear(Wn, m, theta0, Sigma0, sample_std, labeling)
        new(Wn, copy(theta0), copy(Sigma0), sample_std, copy(labeling), similar(theta0), similar(Sigma0))
    end
end

function allocation(policy::GreedyPolicyLinear,Xcurrent,W,X,Y,rng=Random.GLOBAL_RNG)
    return argmax_ties([interact(iw,policy.Wn,Xcurrent,policy.labeling)' * policy.theta_t for iw in 1:policy.Wn],rng)
end