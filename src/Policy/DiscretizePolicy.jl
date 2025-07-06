"""
    DiscretizePolicy{T<:Policy} <: Policy
    DiscretizePolicy(subpolicy::T, FX::Union{CovariatesIndependent,CovariatesCopula}, breakpoints) where {T<:Policy}

Modify `subpolicy` by discretizing the covariates in `FX` before passing them to `subpolicy`.`
"""
mutable struct DiscretizePolicy{T<:Policy} <: Policy
    subpolicy::T
    FX::Union{CovariatesIndependent,CovariatesCopula}
    FX_discretized::Union{CovariatesIndependent,CovariatesCopula}
    gn::Int
    breakpoints::Vector{Vector{Float64}}
    values_discretized::Vector{Vector{Float64}}
    X_discrete::Matrix{Float64}
    function DiscretizePolicy(subpolicy::T, FX::Union{CovariatesIndependent,CovariatesCopula}, breakpoints) where {T<:Policy}
        out = discretizeFX(FX, breakpoints)
        new{T}(deepcopy(subpolicy), FX, out[1], out[3], deepcopy(breakpoints), out[2], Matrix{Float64}(undef, length(FX), 0))
    end
end

function initialize!(policy::DiscretizePolicy, W, X, Y)
    policy.X_discrete = Matrix{Float64}(undef, length(policy.FX), 0)

    X_discrete = similar(X, Float64)
    for i in axes(X, 2)
        X_discrete[:, i] = X_discretize(view(X, :, i), policy)
    end
    initialize!(policy.subpolicy, W, X_discrete, Y)
    return
end

function state_update!(policy::DiscretizePolicy, W, X, Y)
    for i in (size(policy.X_discrete, 2)+1):size(X, 2)
        policy.X_discrete = hcat(policy.X_discrete, X_discretize(view(X, :, i), policy))
    end
    state_update!(policy.subpolicy, W, policy.X_discrete, Y)
    return
end

function implementation(policy::DiscretizePolicy, X_post, W, X, Y)
    X_post_discrete = similar(X_post, Float64)
    for k in axes(X_post, 2)
        X_post_discrete[:, k] = X_discretize(view(X_post, :, k), policy)
    end
    return implementation(policy.subpolicy, X_post_discrete, W, policy.X_discrete, Y)
end

function allocation(policy::DiscretizePolicy, Xcurrent, W, X, Y, rng=Random.default_rng())
    X_current_discrete = X_discretize(Xcurrent, policy)
    return allocation(policy.subpolicy, X_current_discrete, W, policy.X_discrete, Y, rng)
end


function discretizeFX(FX::Union{CovariatesIndependent,CovariatesCopula}, breakpoints)
    @assert length(breakpoints) == length(marginals(FX)) "breakpoints is not the same length as marginals(FX)"
    @assert all([all(x .> 0) for x in diff.(breakpoints)]) "breakpoints are not strictly increasing"
    gn = 1
    discretized_marginals = Vector{Distribution{Univariate,S} where S<:ValueSupport}(undef, length(marginals(FX)))
    values_discretized = Vector{Vector{Float64}}(undef, 0)
    for i in eachindex(marginals(FX))
        m = marginals(FX)[i]
        if typeof(m) <: Categorical || typeof(m) <: OrdinalDiscrete
            gn *= length(support(m))
            discretized_marginals[i] = m
            push!(values_discretized, support(m))
        else
            gn *= length(breakpoints[i]) + 1
            cdfs = cdf.(m, breakpoints[i])
            p = vcat(cdfs[1], diff(cdfs), 1 - cdfs[end])
            v = Vector{Float64}(undef, length(p))
            for j in eachindex(v)
                lb = j > 1 ? breakpoints[i][j-1] : -Inf
                ub = j < length(v) ? breakpoints[i][j] : Inf
                integral, err = quadgk(x -> x * pdf(m, x), lb, ub)
                v[j] = integral / (cdf(m, ub) - cdf(m, lb))
            end
            discretized_marginals[i] = OrdinalDiscrete(v, p)
            push!(values_discretized, v)
        end
    end
    if typeof(FX) <: CovariatesIndependent
        FX_discretized = CovariatesIndependent(discretized_marginals)
    elseif typeof(FX) <: CovariatesCopula
        FX_discretized = CovariatesCopula(discretized_marginals, FX.copula)
    end
    return FX_discretized, values_discretized, gn
end

function discretize(x, breakpoints, values)
    index = 1
    for b in breakpoints
        if x < b
            break
        else
            index += 1
        end
    end
    return values[index]
end

function X_discretize(X, policy::DiscretizePolicy)
    X_discrete = similar(X, Float64)
    index = 1
    for i in eachindex(marginals(policy.FX))
        m = marginals(policy.FX)[i]
        if typeof(m) <: Categorical
            X_discrete[index+1:index+length(support(m))-1] .= X[index+1:index+length(support(m))-1]
            index += length(support(m)) - 1
        elseif typeof(m) <: OrdinalDiscrete
            index += 1
        else
            X_discrete[index+1] = discretize(X[index+1], policy.breakpoints[i], policy.values_discretized[i])
            index += 1
        end
    end
    return X_discrete
end