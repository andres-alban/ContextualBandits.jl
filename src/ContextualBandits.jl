module ContextualBandits
using Random
using LinearAlgebra
using Statistics
import Future.randjump
using Distributed
using Distributions

include("CovariatesGeneration.jl")
using .CovariatesGeneration
export CovariatesCopula, CovariatesIndependent, CovariatesInteracted, OrdinalDiscrete, marginals, covariates_partition

include("utils.jl")
export interact, interact!, interact2!, argmax_ties, argmin_ties, randnMv, labeling2predprog

include("utils_discrete.jl")

include("BayesLinearRegression.jl")
export BayesLinearRegression, BayesUpdateNormal, BayesUpdateNormal!

include("OutcomeModel/OutcomeModel.jl")
export OutcomeModel
include("OutcomeModel/OutcomeLinear.jl")
export OutcomeLinearBayes, OutcomeLinearFixed

include("Policy/Policy.jl")
export Policy
include("policy/PolicyLinear.jl")
export PolicyLinear, RandomPolicyLinear, GreedyPolicyLinear
include("Policy/robust_prior_linear.jl")
export robustify_prior_linear!, default_prior_linear
include("Policy/PolicyLinearDiscrete.jl")
export PolicyLinearDiscrete
include("Policy/KG.jl")
export cKG, iKG
include("Policy/fEVI.jl")
export fEVI, fEVIDiscrete, fEVIDiscreteOnOff
include("Policy/fEVI_MC.jl")
include("Policy/fEVI_MC_without_h.jl")
export fEVI_MC, fEVI_MC_PolicyLinear
include("Policy/ThompsonSampling.jl")
export TSPolicyLinear, TTTSPolicyLinear
include("Policy/ocba.jl")
export OCBAPolicyLinear
include("Policy/biasedcoin.jl")
export BiasedCoinPolicyLinear, RABC_OCBA_PolicyLinear


include("simulation/replication_stochastic.jl")
include("simulation/simulation_stochastic.jl")
include("simulation/SimulationRecorder.jl")
include("simulation/RunningMeanVariance.jl")
export RunningMeanVariance, mean, var, std, mean_stderr
include("simulation/SimulationResults.jl")
export simulation_stochastic, simulation_stochastic_parallel

end # module ContextualBandits
