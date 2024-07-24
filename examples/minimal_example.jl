using ContextualBandits
using Distributions

n = 3
FX = CovariatesIndependent([Normal(),Normal()])
m = length(FX)
T = 10
policies = Dict("random" => RandomPolicy(n))
sample_std = 1.0
mu = rand(n*m)
outcome_model = OutcomeLinear(n,m,mu,sample_std)
reps = 10

results = simulation_stochastic(reps,FX,n,T,policies,outcome_model)

output_random = results["output"]["random"]
keys(output_random)
output_random["regret_on"]
output_random["cumulregret_on"]
output_random["regret_off"]
