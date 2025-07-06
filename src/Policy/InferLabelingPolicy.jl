
"""
    InferLabelingPolicy{T<:PolicyLinear, S<:LabelingSelector} <: Policy
    InferLabelingPolicy(subpolicy, selector, schedule, [labeling0; sigma0, psi, D, z_alpha, c])

Create a policy that modifies `subpolicy` by using a labeling `selector` to infer the labeling
at a specified schedule.
"""
mutable struct InferLabelingPolicy{T<:PolicyLinear,S<:LabelingSelector} <: Policy
    subpolicy::T
    labeling_selector::S
    schedule::Vector{Int}
    labeling0::Vector{Bool}
    sigma0::Float64
    psi::Float64
    D::Matrix{Float64}
    z_alpha::Float64
    c::Float64
    Wpilot::Vector{Int}
    Xpilot::Matrix{Float64}
    Ypilot::Vector{Float64}
    function InferLabelingPolicy(subpolicy::T, selector::S, schedule, labeling0=trues((subpolicy.model.n + 1) * subpolicy.model.m); sigma0=1e6, psi=1, D=fill(Inf, subpolicy.model.n, subpolicy.model.n), z_alpha=2, c=4) where {T<:PolicyLinear,S<:LabelingSelector}
        subpolicy = deepcopy(subpolicy)
        subpolicy.model.labeling .= labeling0
        new{T,S}(subpolicy, selector, schedule, copy(labeling0), sigma0, psi, D, z_alpha, c, zeros(Int, 0), zeros(0, 0), zeros(0))
    end
end

function initialize!(policy::InferLabelingPolicy, W, X, Y)
    policy.subpolicy.model.labeling = policy.labeling0
    theta, Sigma = default_prior_linear(policy.subpolicy.model.n, policy.subpolicy.model.m,
        policy.sigma0, policy.psi, policy.D, policy.subpolicy.model.labeling)
    robustify_prior_linear!(theta, Sigma, policy.subpolicy.model.n, policy.subpolicy.model.m, policy.subpolicy.model.labeling, policy.z_alpha, policy.c)
    policy.subpolicy.model.theta0 = theta
    policy.subpolicy.model.Sigma0 = Sigma
    if length(Y) > 0
        policy.Wpilot = W
        policy.Xpilot = X
        policy.Ypilot = Y
    end
    initialize!(policy.subpolicy, W, X, Y)
    initialize!(policy.labeling_selector)
    return
end

function state_update!(policy::InferLabelingPolicy, W, X, Y)
    index = length(Y)
    if index in policy.schedule && index > 0
        policy.subpolicy.model.labeling .= labeling_selection(policy.labeling_selector, W, X, Y)
        theta, Sigma = default_prior_linear(policy.subpolicy.model.n, policy.subpolicy.model.m,
            policy.sigma0, policy.psi, policy.D, policy.subpolicy.model.labeling)
        robustify_prior_linear!(theta, Sigma, policy.subpolicy.model.n, policy.subpolicy.model.m, policy.subpolicy.model.labeling, policy.z_alpha, policy.c)
        policy.subpolicy.model.theta0 = theta
        policy.subpolicy.model.Sigma0 = Sigma
        initialize!(policy.subpolicy, policy.Wpilot, policy.Xpilot, policy.Ypilot)
        for i in 1:index
            state_update!(policy.subpolicy, W, X, view(Y, 1:i))
        end
    else
        state_update!(policy.subpolicy, W, X, Y)
    end
    return
end

function implementation(policy::InferLabelingPolicy, X_post, W, X, Y)
    return implementation(policy.subpolicy, X_post, W, X, Y)
end

function allocation(policy::InferLabelingPolicy, Xcurrent, W, X, Y, rng=Random.default_rng())
    return allocation(policy.subpolicy, Xcurrent, W, X, Y, rng)
end

function policy_labeling(policy::InferLabelingPolicy)
    return policy_labeling(policy.subpolicy)
end
