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
include("OutcomeModel/OutcomeLinear.jl")
export OutcomeLinearBayes, OutcomeLinearFixed

include("Policy/Policy.jl")
export Policy
include("Policy/PolicyLinear.jl")
export PolicyLinear, RandomPolicyLinear, GreedyPolicyLinear
include("Policy/robust_prior_linear.jl")
export robustify_prior_linear!, default_prior_linear
include("Policy/PolicyLinearDiscrete.jl")
export PolicyLinearDiscrete
include("Policy/KG.jl")
export cKG, iKG
include("Policy/fEVI.jl")
export fEVI, fEVIDiscrete, fEVIDiscreteOnOff


include("simulation/replication_stochastic.jl")
include("simulation/simulation_stochastic.jl")
include("simulation/SimulationRecorder.jl")
include("simulation/RunningMeanVariance.jl")
export RunningMeanVariance, mean, var, std, mean_stderr
include("simulation/SimulationResults.jl")
export simulation_stochastic, simulation_stochastic_parallel

end # module ContextualBandits
