module ContextualBandits
using Random
using LinearAlgebra
using Statistics
import Future.randjump
using Distributed
using Distributions

include("utils.jl")
export interact, interact!, interact2!, BayesUpdateNormal, BayesUpdateNormal!, argmax_ties, argmin_ties, randnMv

include("CovariatesGeneration.jl")
using .CovariatesGeneration
export CovariatesCopula, CovariatesIndependent, CovariatesInteracted, OrdinalDiscrete, marginals

include("utils_discrete.jl")

include("OutcomeModel/OutcomeModel.jl")
export OutcomeModel
include("OutcomeModel/OutcomeLabel.jl")
export LinearOutcomeLabelRandom, LinearOutcomeLabelFixed

include("Policy/Policy.jl")
export Policy
include("Policy/PolicyLabel.jl")
export PolicyLabel, RandomPolicyLabel, GreedyPolicyLabel

include("simulation/replication_stochastic.jl")
include("simulation/simulation_stochastic.jl")
include("simulation/SimulationRecorder.jl")
include("simulation/RunningMeanVariance.jl")
include("simulation/SimulationResults.jl")
export simulation_stochastic, simulation_stochastic_parallel

end # module ContextualBandits
