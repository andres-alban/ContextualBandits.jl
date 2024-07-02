"""
    ocba_normal(mu,sample_var)

Compute the allocation probabilities prescribed by the OCBA algorithm
for a normal distribution with mean `mu` and sample variance `sample_var`.
[Chen, C., Chick, S.E., Lee, L.H., & Pujowidianto, N.A. (2015). Ranking and Selection: Efficient Simulation Budget Allocation.](https://doi.org/10.1007/978-1-4939-1384-8_3)
"""
function ocba_normal(mu,sample_var)
    length(mu) == length(sample_var) || throw(ArgumentError("mu and sample_var must be the same length"))
    imax = argmax(mu)
    imin = argmin(mu)
    mumax = mu[imax]
    mumin = mu[imin]
    if mumax == mumin
        return ones(length(mu)) / length(mu)
    end
    p = zeros(length(mu))
    p[imin] = 1.0

    for i in eachindex(p)
        if i == imax || i == imin
            continue
        end
        p[i] = sample_var[i] / (mu[i]-mumax)^2 / (sample_var[imin] / (mumin - mumax)^2)
    end
    pnotmax = p[p .!= 0]
    varnotmax = sample_var[p .!= 0]
    p[imax] = sqrt(sample_var[imax]) * sqrt(sum(pnotmax.^2 ./ varnotmax))
    p /= sum(p)
    return p
end

"""
    OCBAPolicyLinear <: PolicyLinear
    OCBAPolicyLinear(Wn, m, theta0, Sigma0, sample_std, predictive, labeling=vcat(falses(m),trues(Wn*m)))
    OCBAPolicyLinear(Wn, m, theta0, Sigma0, sample_std, FX::Union{CovariatesCopula, CovariatesIndependent}, labeling=vcat(falses(m),trues(Wn*m)))

Allocate treatment using the OCBA algorithm and update based on the linear model with labeling to make an implementation.

`predictive` is a vector of integers that specifies the covariates that define predictive groups,
i.e., patients in the same predictive group have the same covariate values for the covariates specified in `predictive`.
Instead of predictive, you can pass `FX`, which is a `CovariatesCopula` or `CovariatesIndependent` object,
and the predictive groups will be automatically determined based on the `labeling`.

> NOTE: the covariates specified in predictive should be discrete.
> If it is not discrete, the algorithm will run but the results may not be meaningful.
"""
struct OCBAPolicyLinear <: PolicyLinear 
    model::BayesLinearRegression
    predictive::Vector{Int}
    function OCBAPolicyLinear(model, predictive)
        all(predictive .∈ Ref(1:model.m)) || throw(ArgumentError("predictive must be a subset of 1:model.m"))
        new(model, predictive)
    end
end

function OCBAPolicyLinear(Wn, m, theta0, Sigma0, sample_std, predictive, labeling=vcat(falses(m),trues(Wn*m)))
    OCBAPolicyLinear(BayesLinearRegression(Wn, m, theta0, Sigma0, sample_std, labeling), predictive)
end

function OCBAPolicyLinear(Wn, m, theta0, Sigma0, sample_std, FX::Union{CovariatesCopula, CovariatesIndependent}, labeling=vcat(falses(m),trues(Wn*m)))
    predictive, _ = labeling2predprog(Wn, FX, labeling)
    OCBAPolicyLinear(BayesLinearRegression(Wn, m, theta0, Sigma0, sample_std, labeling), predictive)
end

function allocation(policy::OCBAPolicyLinear,Xcurrent,W,X,Y,rng=Random.GLOBAL_RNG)
    # We only consider patients in the same predictive group
    index_subgroup = [X[policy.predictive,i] == Xcurrent[policy.predictive] for i in axes(X,2)]
    W_subgroup = @view W[index_subgroup]
    p = [sum(W_subgroup .== w) for w in 1:policy.model.Wn]
    unobserved_treatments = findall(p .== 0)
    if length(unobserved_treatments) == 1
        return unobserved_treatments[1]
    elseif length(unobserved_treatments) > 1
        return rand(rng,unobserved_treatments)
    end
    p /= length(W_subgroup)
    means = [interact(iw,policy.model.Wn,Xcurrent, policy.model.labeling)' * policy.model.theta_t for iw in 1:policy.model.Wn]
    # Before computing the OCBA probabilities, we would like to identify the alternatives 
    # that are equivalent in the sense that they are perfectly correlated and 
    # have the same mean and variance. For simplicity, here we assume that arms
    # with same mean are equivalent. Those alternatives that are equivalent will
    # be passed as a single alternative to the OCBA algorithm and the probability 
    # of the OCBA algorithm will be evenly distributed among equivalent arms.
    means,m,c = unique_values_helper(means)
    vars = ones(length(means)) * policy.model.sample_std^2
    p_ocba = ocba_normal(means,vars)
    p_ocba_final = [p_ocba[m[i]]/c[i] for i in 1:policy.model.Wn]
    return argmax_ties(p_ocba_final - p, rng)
end

"""
    unique_values_helper(x)

Identify the unique values in vector `x` and additionally return a mapping from 
the original vector to the unique values and the counts of each unique value.
"""
function unique_values_helper(x)
    u = unique(x)
    mapping = Vector{Int}(undef,length(x))
    count_unique = Vector{Int}(undef,length(x))
    for i in eachindex(u)
        same = isequal.(x, u[i])
        mapping[same] .= i
        count_unique[same] .= sum(same)
    end
    return u, mapping, count_unique
end