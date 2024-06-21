module ContextualBandits
using Random
using LinearAlgebra

include("utils.jl")
export interact, interact!, interact2!, BayesUpdateNormal, BayesUpdateNormal!, argmax_ties, argmin_ties

include("CovariatesGeneration.jl")
using .CovariatesGeneration
export CovariatesCopula, CovariatesIndependent, CovariatesInteracted, OrdinalDiscrete, marginals

include("OutcomeModel/OutcomeModel.jl")
export OutcomeModel
include("OutcomeModel/OutcomeLabel.jl")
export LinearOutcomeLabelRandom, LinearOutcomeLabelFixed

include("Policy/Policy.jl")
export Policy
include("Policy/PolicyLabel.jl")
export PolicyLabel, RandomPolicyLabel, GreedyPolicyLabel


end # module ContextualBandits
