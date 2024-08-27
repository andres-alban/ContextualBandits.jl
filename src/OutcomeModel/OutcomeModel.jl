"""
    OutcomeModel

Supertype for models that determine mean outcomes and noisy outcomes.

The functions [`outcome_model_state!`](@ref), [`mean_outcome`](@ref), [`noisy_outcome`](@ref), and  [`noise_outcome`](@ref)
take subtypes of `OutcomeModel` as the first argument.
Each subtype should implement these functions.
"""
abstract type OutcomeModel end

"""
    outcome_model_state!(outcome_model::OutcomeModel,rng::AbstractRNG=Random.default_rng())

Set the state of an outcome model. Mainly used to change the state of random instances.
"""
function outcome_model_state!(outcome_model::OutcomeModel, rng::AbstractRNG=Random.default_rng())
    return
end

"""
    mean_outcome(outcome_model::OutcomeModel,W,X)

Compute the mean outcome of treatment `W` with covariates `X`.
"""
function mean_outcome(outcome_model::OutcomeModel, W, X)
    return 0.0
end

"""
    noisy_outcome(outcome_model::OutcomeModel,W,X,Z)

Compute noisy outcome of treatment `W` with covariates `X` given noise `Z`. 
`Z` is generated with [`noise_outcome`](@ref).

Usually calls [`mean_outcome`](@ref) and adds zero mean noise `Z`.
"""
function noisy_outcome(outcome_model::OutcomeModel, W, X, Z)
    return 0.0
end

"""
    noise_outcome(outcome_model::OutcomeModel,rng::AbstractRNG=Random.default_rng())

Generate noise object to be passed to [noisy_outcome](@ref).

It can generate a normal distribution with zero mean for Gaussian models or vectors of random variables for more complex models.
"""
function noise_outcome(outcome_model::OutcomeModel, rng::AbstractRNG=Random.default_rng())
    return 0.0
end