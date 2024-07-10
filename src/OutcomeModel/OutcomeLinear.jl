"""
    OutcomeLinearBayes <: OutcomeModel
    OutcomeLinearBayes(n, m, theta0, Sigma0, sample_std[, labeling])

Linear outcome model with coefficients randomly drawn from a normal prior distribution
with mean vector `theta0` and covariance matrix `Sigma0`.
`n` is the number of treatments, `m` is the number of covariates, including the intercept term.
The first `m` coefficients represent prognostic effects (effects independent of the treatment)
for each of the `m` covariates.
The following `m` coefficients represent the predictive effects of treatment 1
(interaction between treatment and covariates).
In total, there are at most `(n+1)*m` coefficients.
`labeling` is a boolean vector of length `(n+1)*m` indicationg active coefficients. If any coefficient
is knwon to be zero (inactive), the corresponding entry of `labeling` should be `false`.
The length of `theta0` and the dimensions of `Sigma0` should be equal to the number of active coefficients=`sum(labeling)`.
By default, only predictive coefficients are active.

Outcomes are observed with white noise around the mean with sample standard deviation `sample_std`.
"""
struct OutcomeLinearBayes <: OutcomeModel
    n::Int
    m::Int # number of covariates, including the intercept term
    theta0::Vector{Float64}
    Sigma0::Array{Float64,2}
    sample_std::Float64
    labeling::BitVector
    mu::Vector{Float64}
    function OutcomeLinearBayes(n, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m),trues(n*m)))
        d = sum(labeling)
        length(labeling) == (n+1)*m || throw(DomainError(labeling,"`labeling` must have length `(n+1)*m`."))
        length(theta0) == d || throw(DomainError(theta0,"`theta0` must be of length `sum(labeling)`."))
        size(Sigma0) == (d,d) || throw(DomainError(Sigma0,"`Sigma0` must be of dimensions `sum(labeling)`."))
        issymmetric(Sigma0) || throw(DomainError(Sigma0,"`Sigma0` must be symmetric."))
        minimum(eigvals(Sigma0)) >= 0 || throw(DomainError(Sigma0,"`Sigma0` must be positive semidefinite."))
        mu = zeros(d)
        new(n, m, copy(theta0), copy(Sigma0), sample_std, copy(labeling), mu)
    end
end

function outcome_model_state!(outcome_model::OutcomeLinearBayes,rng::AbstractRNG=Random.GLOBAL_RNG)
    outcome_model.mu .= randnMv(rng, outcome_model.theta0, outcome_model.Sigma0)
    return
end

function mean_outcome(outcome_model::OutcomeLinearBayes,W,X)
    return interact(W,outcome_model.n,X,outcome_model.labeling)' * outcome_model.mu
end

function noisy_outcome(outcome_model::OutcomeLinearBayes,W,X,Z)
    return mean_outcome(outcome_model,W,X) + outcome_model.sample_std*Z
end

function noise_outcome(outcome_model::OutcomeLinearBayes,rng::AbstractRNG=Random.GLOBAL_RNG)
    return randn(rng)
end

###############################################################################

"""
    OutcomeLinear <: OutcomeModel
    OutcomeLinear(n, m, mu, sample_std[, labeling])

Linear outcome model with fixed coefficients `mu`.
`n` is the number of treatments, `m` is the number of covariates, including the intercept term.
The first `m` coefficients represent prognostic effects (effects independent of the treatment)
for each of the `m` covariates.
The following `m` coefficients represent the predictive effects of treatment 1
(interaction between treatment and covariates).
In total, there are at most `(n+1)*m` coefficients.
`labeling` is a boolean vector of length `(n+1)*m` indicationg active coefficients. If any coefficient
is knwon to be zero (inactive), the corresponding entry of `labeling` should be `false`.
The length of `mu` should be equal to the number of active coefficients=`sum(labeling)`.
By default, only predictive coefficients are active.

Outcomes are observed with white noise around the mean with sample standard deviation `sample_std`.
"""
struct OutcomeLinear <: OutcomeModel
    n::Int
    m::Int # number of covariates, including the intercept term
    mu::Vector{Float64}
    sample_std::Float64
    labeling::BitVector
    function OutcomeLinear(n, m, mu, sample_std, labeling=vcat(falses(m),trues(n*m)))
        d = sum(labeling)
        length(labeling) == (n+1)*m || throw(DomainError(labeling,"`labeling` must have length `(n+1)*m`."))
        length(mu) == d || throw(DomainError(mu,"`mu` must be of length `sum(labeling)`."))
        new(n, m, copy(mu), sample_std, copy(labeling))
    end
end

function outcome_model_state!(outcome_model::OutcomeLinear,rng::AbstractRNG=Random.GLOBAL_RNG)
    return
end

function mean_outcome(outcome_model::OutcomeLinear,W,X)
    return interact(W,outcome_model.n,X,outcome_model.labeling)' * outcome_model.mu
end

function noisy_outcome(outcome_model::OutcomeLinear,W,X,Z)
    return mean_outcome(outcome_model,W,X) + outcome_model.sample_std*Z
end

function noise_outcome(outcome_model::OutcomeLinear,rng::AbstractRNG=Random.GLOBAL_RNG)
    return randn(rng)
end