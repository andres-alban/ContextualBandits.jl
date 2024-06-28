"""
    OutcomeLinearBayes <: OutcomeModel
    OutcomeLinearBayes(Wn, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m),trues(Wn*m)))

Linear outcome model with coefficients randomly drawn from a normal prior distribution
with mean vector `theta0` and covariance matrix `Sigma0`.
`Wn` is the number of treatments, `m` is the number of covariates, including the intercept term.
The first `m` coefficients represent prognostic effects (effects independent of the treatment)
for each of the `m` covariates.
The following `m` coefficients represent the predictive effects of treatment 1
(interaction between treatment and covariates).
In total, there are at most `(Wn+1)*m` coefficients.
`labeling` is a boolean vector of length `(Wn+1)*m` indicationg active coefficients. If any coefficient
is knwon to be zero (inactive), the corresponding entry of `labeling` should be `false`.
The length of `theta0` and the dimensions of `Sigma0` should be equal to the number of active coefficients=`sum(labeling)`.
By default, only predictive coefficients are active.

Outcomes are observed with white noise around the mean with sample standard deviation `sample_std`.
"""
struct OutcomeLinearBayes <: OutcomeModel
    Wn::Int
    m::Int # number of covariates, including the intercept term
    theta0::Vector{Float64}
    Sigma0::Array{Float64,2}
    sample_std::Float64
    labeling::BitVector
    mu::Vector{Float64}
    function OutcomeLinearBayes(Wn, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m),trues(Wn*m)))
        d = sum(labeling)
        length(labeling) == (Wn+1)*m || throw(DomainError(labeling,"`labeling` must have length `(Wn+1)*m`."))
        length(theta0) == d || throw(DomainError(theta0,"`theta0` must be of length `sum(labeling)`."))
        size(Sigma0) == (d,d) || throw(DomainError(Sigma0,"`Sigma0` must be of dimensions `sum(labeling)`."))
        issymmetric(Sigma0) || throw(DomainError(Sigma0,"`Sigma0` must be symmetric."))
        minimum(eigvals(Sigma0)) >= 0 || throw(DomainError(Sigma0,"`Sigma0` must be positive semidefinite."))
        mu = zeros(d)
        new(Wn, m, copy(theta0), copy(Sigma0), sample_std, copy(labeling), mu)
    end
end

function outcome_model_state!(outcome_model::OutcomeLinearBayes,rng::AbstractRNG=Random.GLOBAL_RNG)
    outcome_model.mu .= randnMv(rng, outcome_model.theta0, outcome_model.Sigma0)
    return
end

function mean_outcome(outcome_model::OutcomeLinearBayes,W,X)
    return interact(W,outcome_model.Wn,X,outcome_model.labeling)' * outcome_model.mu
end

function noisy_outcome(outcome_model::OutcomeLinearBayes,W,X,Z)
    return mean_outcome(outcome_model,W,X) + outcome_model.sample_std*Z
end

function noise_outcome(outcome_model::OutcomeLinearBayes,rng::AbstractRNG=Random.GLOBAL_RNG)
    return randn(rng)
end

###############################################################################

"""
    OutcomeLinearFixed <: OutcomeModel
    OutcomeLinearFixed(Wn, m, mu, sample_std, labeling=vcat(falses(m),trues(Wn*m)))

Linear outcome model with fixed coefficients `mu`.
`Wn` is the number of treatments, `m` is the number of covariates, including the intercept term.
The first `m` coefficients represent prognostic effects (effects independent of the treatment)
for each of the `m` covariates.
The following `m` coefficients represent the predictive effects of treatment 1
(interaction between treatment and covariates).
In total, there are at most `(Wn+1)*m` coefficients.
`labeling` is a boolean vector of length `(Wn+1)*m` indicationg active coefficients. If any coefficient
is knwon to be zero (inactive), the corresponding entry of `labeling` should be `false`.
The length of `mu` should be equal to the number of active coefficients=`sum(labeling)`.
By default, only predictive coefficients are active.

Outcomes are observed with white noise around the mean with sample standard deviation `sample_std`.
"""
struct OutcomeLinearFixed <: OutcomeModel
    Wn::Int
    m::Int # number of covariates, including the intercept term
    mu::Vector{Float64}
    sample_std::Float64
    labeling::BitVector
    function OutcomeLinearFixed(Wn, m, mu, sample_std, labeling=vcat(falses(m),trues(Wn*m)))
        d = sum(labeling)
        length(labeling) == (Wn+1)*m || throw(DomainError(labeling,"`labeling` must have length `(Wn+1)*m`."))
        length(mu) == d || throw(DomainError(mu,"`mu` must be of length `sum(labeling)`."))
        new(Wn, m, copy(mu), sample_std, copy(labeling))
    end
end

function outcome_model_state!(outcome_model::OutcomeLinearFixed,rng::AbstractRNG=Random.GLOBAL_RNG)
    return
end

function mean_outcome(outcome_model::OutcomeLinearFixed,W,X)
    return interact(W,outcome_model.Wn,X,outcome_model.labeling)' * outcome_model.mu
end

function noisy_outcome(outcome_model::OutcomeLinearFixed,W,X,Z)
    return mean_outcome(outcome_model,W,X) + outcome_model.sample_std*Z
end

function noise_outcome(outcome_model::OutcomeLinearFixed,rng::AbstractRNG=Random.GLOBAL_RNG)
    return randn(rng)
end