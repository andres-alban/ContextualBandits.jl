"""
    struct LinearOutcomeLabelRandom <: OutcomeModel
    LinearOutcomeLabelRandom(Wn, m, labels, theta0, Sigma0, sample_std)

Linear outcome model with coefficients randomly drawn from a normal prior distribution. White noise around the mean.
`labels` is the vector of labels for active coefficients.
"""
struct LinearOutcomeLabelRandom <: OutcomeModel
    Wn::Int
    m::Int # number of covariates, including the intercept term
    labels::BitVector
    theta0::Vector{Float64}
    Sigma0::Array{Float64,2}
    sample_std::Float64
    mu::Vector{Float64}
    function LinearOutcomeLabelRandom(Wn, m, labels, theta0, Sigma0, sample_std)
        d = sum(labels)
        length(labels) == (Wn+1)*m || throw(DomainError(labels,"`labels` must have length `(Wn+1)*m`."))
        length(theta0) == d || throw(DomainError(theta0,"`theta0` must be of length `sum(labels)`."))
        size(Sigma0) == (d,d) || throw(DomainError(Sigma0,"`Sigma0` must be of dimensions `sum(labels)`."))
        issymmetric(Sigma0) || throw(DomainError(Sigma0,"`Sigma0` must be symmetric."))
        minimum(eigvals(Sigma0)) >= 0 || throw(DomainError(Sigma0,"`Sigma0` must be positive semidefinite."))
        mu = zeros(d)
        new(Wn, m, copy(labels), copy(theta0), copy(Sigma0), sample_std, mu)
    end
end

function outcome_model_state!(outcome_model::LinearOutcomeLabelRandom,rng::AbstractRNG=Random.GLOBAL_RNG)
    # The following is a sample draw from a multivariate normal distribution that allows for a positive semidefinite covariance matrix.
    # The MvNormal distribution in Distributions.jl requires a positive definite covariance matrix.
    # The constructor of LinearOutcomeLabelRandom ensures that Sigma0 is positive semidefinite.
    chol = cholesky(outcome_model.Sigma0,RowMaximum(),check=false)
    outcome_model.mu .= outcome_model.theta0 + chol.L[invperm(chol.p),1:chol.rank]*randn(rng,chol.rank)
    return
end

function mean_outcome(outcome_model::LinearOutcomeLabelRandom,W,X)
    return interact(W,outcome_model.Wn,X,outcome_model.labels)' * outcome_model.mu
end

function noisy_outcome(outcome_model::LinearOutcomeLabelRandom,W,X,Z)
    return mean_outcome(outcome_model,W,X) + outcome_model.sample_std*Z
end

function noise_outcome(outcome_model::LinearOutcomeLabelRandom,rng::AbstractRNG=Random.GLOBAL_RNG)
    return randn(rng)
end

###############################################################################

"""
    struct LinearOutcomeLabelFixed <: OutcomeModel
    LinearOutcomeLabelFixed(Wn, m, labels, sample_std, mu)

Linear outcome model with fixed coefficients `mu` and white noise around the mean.
`labels` is the vector of labels for active coefficients.
"""
struct LinearOutcomeLabelFixed <: OutcomeModel
    Wn::Int
    m::Int # number of covariates, including the intercept term
    labels::BitVector
    sample_std::Float64
    mu::Vector{Float64}
    function LinearOutcomeLabelFixed(Wn, m, labels, sample_std, mu)
        d = sum(labels)
        length(labels) == (Wn+1)*m || throw(DomainError(labels,"`labels` must have length `(Wn+1)*m`."))
        length(mu) == d || throw(DomainError(mu,"`mu` must be of length `sum(labels)`."))
        new(Wn, m, copy(labels), sample_std, copy(mu))
    end
end

function outcome_model_state!(outcome_model::LinearOutcomeLabelFixed,rng::AbstractRNG=Random.GLOBAL_RNG)
    return
end

function mean_outcome(outcome_model::LinearOutcomeLabelFixed,W,X)
    return interact(W,outcome_model.Wn,X,outcome_model.labels)' * outcome_model.mu
end

function noisy_outcome(outcome_model::LinearOutcomeLabelFixed,W,X,Z)
    return mean_outcome(outcome_model,W,X) + outcome_model.sample_std*Z
end

function noise_outcome(outcome_model::LinearOutcomeLabelFixed,rng::AbstractRNG=Random.GLOBAL_RNG)
    return randn(rng)
end