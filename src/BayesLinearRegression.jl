"""
    BayesLinearRegression(n, m, theta0, Sigma0, sample_std[, labeling])

Bayesian linear regression model with `n` treatments and `m` covariates.

See also: [ContextualBandits.initialize!](@ref), [ContextualBandits.state_update!](@ref),
[BayesUpdateNormal](@ref), [BayesUpdateNormal!](@ref)
"""
mutable struct BayesLinearRegression
    n::Int
    m::Int
    theta0::Vector{Float64}
    Sigma0::Matrix{Float64}
    sample_std::Float64
    labeling::BitVector
    theta_t::Vector{Float64}
    Sigma_t::Matrix{Float64}
    function BayesLinearRegression(n, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m), trues(n * m)))
        checkInputPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
        new(n, m, copy(theta0), copy(Sigma0), sample_std, copy(labeling), similar(theta0), similar(Sigma0))
    end
end

"""
    initialize!(model::BayesLinearRegression)

Reset the posterior mean `model.theta_t` and covariance matrix `model.Sigma_t` to the prior 
mean `theta0` and covariance matrix `Sigma0`.
"""
function initialize!(model::BayesLinearRegression)
    model.theta_t = copy(model.theta0)
    model.Sigma_t = copy(model.Sigma0)
    return
end

"""
    state_update!(model::BayesLinearRegression, W, X, Y)

Update the posterior mean `model.theta_t` and covariance matrix `model.Sigma_t`
after observing treatments `W`, covariates `X`, and outcomes `Y`.
"""
function state_update!(model::BayesLinearRegression, W, X, Y)
    BayesUpdateNormal!(model.theta_t, model.Sigma_t, interact(W, model.n, X, model.labeling), Y, model.sample_std)
    return
end

function model_labeling(model::BayesLinearRegression)
    return model.labeling
end

function checkInputPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    d = sum(labeling)
    length(labeling) == (n + 1) * m || throw(DomainError(labeling, "`labeling` must have length `(n+1)*m`."))
    length(theta0) == d || throw(DomainError(theta0, "`theta0` must be of length `sum(labeling)`."))
    size(Sigma0) == (d, d) || throw(DomainError(Sigma0, "`Sigma0` must be of dimensions `sum(labeling)`."))
    issymmetric(Sigma0) || throw(DomainError(Sigma0, "`Sigma0` must be symmetric."))
    mineigval = minimum(eigvals(Sigma0))
    mineigval >= -sqrt(eps(Float64)) || throw(DomainError(Sigma0, "`Sigma0` must be positive semidefinite."))
    mineigval > 0 || @warn "`Sigma0` is semi-definite. Numerical errors are possible."
    sample_std >= 0 || throw(DomainError(sample_std, "`sample_std` must be positive."))
    return true
end



"""
    BayesUpdateNormal(theta,Sigma,X,y,sample_std)

Update Bayesian hyperparameters `theta` (mean vector) and `Sigma` (covariance matrix) of a linear regression model
with regressor matrix `X`, outputs `y`, and sampling standard deviation `sample_std`.

Return a copy of the updated `theta` and `Sigma`.

`X` can be a vector of regressors or a matrix of regressors, where each column is a vector of regressors.
In the latter case, `y` and `sample_std` can be vectors of the same length as the number of columns of `X`.

See also: [BayesUpdateNormal!](@ref)
"""
function BayesUpdateNormal(theta, Sigma, X, y, sample_std)
    BayesUpdateNormal!(copy(theta), copy(Sigma), X, y, sample_std)
end

"""
    BayesUpdateNormal!(theta,Sigma,X,y,sample_std)

In-place version of [BayesUpdateNormal](@ref)
"""
function BayesUpdateNormal!(theta, Sigma, X, y, sample_std)
    theta .= theta .+ (y .- X' * theta) ./ (sample_std^2 .+ X' * Sigma * X) .* Sigma * X
    Sigma .= Sigma - ((X' * Sigma)' * (X' * Sigma)) ./ (sample_std^2 .+ X' * Sigma * X)
    return theta, Sigma
end

function BayesUpdateNormal!(theta, Sigma, X, y::AbstractVector, sample_std)
    for i in eachindex(y)
        BayesUpdateNormal!(theta, Sigma, view(X, :, i), y[i], sample_std)
    end
    return theta, Sigma
end

function BayesUpdateNormal!(theta, Sigma, X, y::AbstractVector, sample_std::AbstractVector)
    for i in eachindex(y)
        BayesUpdateNormal!(theta, Sigma, view(X, :, i), y[i], sample_std[i])
    end
    return theta, Sigma
end



mutable struct BayesLinearRegressionDiscrete
    n::Int
    m::Int
    theta0::Vector{Float64}
    Sigma0::Matrix{Float64}
    sample_std::Float64
    labeling::BitVector
    FX::Union{CovariatesIndependent,CovariatesCopula}
    gn::Int
    p::Vector{Float64}
    theta_t::Vector{Float64}
    Sigma_t::Matrix{Float64}
    function BayesLinearRegressionDiscrete(n, m, theta0, Sigma0, sample_std, FX, labeling=vcat(falses(m), trues(n * m)))
        checkInputPolicyLinearDiscrete(n, m, theta0, Sigma0, sample_std, labeling, FX)
        gn = total_groups(FX)
        new(n, m, copy(theta0), copy(Sigma0), sample_std, copy(labeling), FX, gn, X2g_probs(FX), similar(theta0), similar(Sigma0))
    end
end

function initialize!(model::BayesLinearRegressionDiscrete, W=Int[], X=Float64[], Y=Float64[])
    if length(Y) > 0
        WX = interact(W, model.n, X, model.labeling)
        theta, Sigma = BayesUpdateNormal(model.theta0, model.Sigma0, WX, Y, model.sample_std)
        robustify_prior_linear!(theta, Sigma, model.n, model.m, model.labeling)
        model.theta_t, model.Sigma_t = X2g_prior(theta, Sigma, model.FX, model.labeling, model.n)
    else
        model.theta_t, model.Sigma_t = X2g_prior(model.theta0, model.Sigma0, model.FX, model.labeling, model.n)
    end
    return
end

function state_update!(model::BayesLinearRegressionDiscrete, W, X, Y)
    if W isa AbstractVector
        index = [treatment_g2index(W[i], X2g(view(X, :, i), model.FX), model.gn) for i in eachindex(W)]
    else
        index = treatment_g2index(W, X2g(X, model.FX), model.gn)
    end
    BayesUpdateNormalDiscrete!(model.theta_t, model.Sigma_t, index, Y, model.sample_std)
    return
end

function model_labeling(model::BayesLinearRegressionDiscrete)
    return model.labeling
end

function checkInputPolicyLinearDiscrete(n, m, theta0, Sigma0, sample_std, labeling, FX)
    m == length(FX) || throw(DomainError("The number of covariates m must be the same as the length of FX."))
    checkInputPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling)
    for marginal in marginals(FX)
        typeof(marginal) <: Categorical || typeof(marginal) <: OrdinalDiscrete || throw(DomainError("marginal distributions need to be either `Categorical` or `OrdinalDiscrete`."))
    end
end

"""
    BayesUpdateNormalDiscrete(theta,Sigma,index,y,sample_std)

Same as [BayesUpdateNormal](@ref) but for discrete models, where each entry of
`theta` corresponds to an alternative and `index` is the sampled alternative.

```julia
X=zeros(length(theta))
X[index] = 1
BayesUpdateNormal(theta,Sigma,X,y,sample_std) == BayesUpdateNormalDiscrete(theta,Sigma,index,y,sample_std)
```
"""
function BayesUpdateNormalDiscrete(theta, Sigma, index, y, sample_std)
    BayesUpdateNormalDiscrete!(copy(theta), copy(Sigma), index, y, sample_std)
end

"""
    BayesUpdateNormalDiscrete!(theta,Sigma,index,y,sample_std)

In-place version of [BayesUpdateNormalDiscrete](@ref).
"""
function BayesUpdateNormalDiscrete!(theta, Sigma, index, y, sample_std)
    theta .= theta .+ (y .- theta[index]) ./ (sample_std^2 .+ Sigma[index, index]) .* Sigma[:, index]
    Sigma .= Sigma - (Sigma[:, index] * Sigma[:, index]') ./ (sample_std^2 .+ Sigma[index, index])
    return theta, Sigma
end

function BayesUpdateNormalDiscrete!(theta, Sigma, index, y::AbstractVector, sample_std)
    for i in eachindex(y)
        BayesUpdateNormalDiscrete!(theta, Sigma, index[i], y[i], sample_std)
    end
    return theta, Sigma
end

function BayesUpdateNormalDiscrete!(theta, Sigma, index, y::AbstractVector, sample_std::AbstractVector)
    for i in eachindex(y)
        BayesUpdateNormalDiscrete!(theta, Sigma, index[i], y[i], sample_std[i])
    end
    return theta, Sigma
end