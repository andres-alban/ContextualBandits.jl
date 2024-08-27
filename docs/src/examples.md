Examples
========

## Alban, Chick, Zoumpoulis (2024)
The following example recreates the results of Figure 2a in [Alban, Chick, Zoumpoulis (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4160045). It requires two additional packages to save the data and plot it:
```
pkg> add JLD2, Plots
```

### Example simulation

The simulation can take several hours because it runs 5000 replications (`reps=5000`). With ten cores on our machines, this simulation can run in about an hour. However, it may run faster or slower in your machine. We recommend you try it first setting `reps=10` or another small number, so you can predict how long it will take to run. You should also adjust the number of cores depending on your computer capacity.

```julia
using Distributed
addprocs(10) # adjust the number of cores that will be used for simulation
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using ContextualBandits
using Random
using Distributions
using Statistics
using JLD2
using LinearAlgebra

# Setting up model
n = 8 # number of treatments
p = [59, 129, 184, 150] ./ 522 # 4 categories
p_prog = [0.25, 0.5, 0.25] # low, medium, high
FX = CovariatesIndependent([Categorical(p), OrdinalDiscrete(p_prog), OrdinalDiscrete(p_prog)])
FXtilde = CovariatesIndependent([Categorical(p), OrdinalDiscrete(p_prog), OrdinalDiscrete(p_prog)])
m = length(FX)
sample_std = 1.0
labeling = BitVector(
    [1, 1, 1, 1, 1, 0, # prognostic
    0, 0, 0, 0, 0, 0, # treatment 1
    0, 0, 0, 0, 0, 0, # treatment 2
    0, 0, 0, 0, 0, 0, # treatment 3
    1, 0, 0, 0, 0, 0, # treatment 4
    0, 1, 0, 0, 0, 0, # treatment 5
    0, 1, 0, 0, 0, 0, # treatment 6
    0, 1, 0, 0, 0, 0, # treatment 7
    0, 1, 0, 0, 0, 0, # treatment 8
])

# Setting up nature's distribution
theta_nat = zeros(sum(labeling))
Sigma_nat = 4 * diagm(ones(sum(labeling)))
# Positive correlation for the predictive coefficients
Sigma_nat[7:10, 7:10] = [4 1 1 0.5;
    1 4 0.5 1;
    1 0.5 4 1;
    0.5 1 1 4]

outcome_model = OutcomeLinearBayes(n, m, theta_nat, Sigma_nat, sample_std, labeling)
delay = 0

# Setting up prior
labeling0 = labeling
sigma0 = 2
psi = log(2)
D = [
    0 2 2 3 2 3 3 Inf;
    2 0 3 2 3 2 Inf 3;
    2 3 0 2 3 Inf 2 3;
    3 2 2 0 Inf 3 3 2;
    2 3 3 Inf 0 2 2 3;
    3 2 Inf 3 2 0 3 2;
    3 Inf 2 3 2 3 0 2;
    Inf 3 3 2 3 2 2 0
]
theta0, Sigma0 = default_prior_linear(n, m, sigma0, psi, D, labeling0)
robustify_prior_linear!(theta0, Sigma0, n, m, labeling0)

## Random policy
random_policy = RandomPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling0)

## Thompson sampling
TS_policy = TSPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling0)

## Top-two Thompson sampling
beta = 0.5
maxiter = 100
TTTS_policy = TTTSPolicyLinear(n, m, theta0, Sigma0, sample_std, beta, maxiter, labeling0)

## fEVI policy
fEVI_policy = fEVIDiscrete(n, m, theta0, Sigma0, sample_std, FX, labeling0)

## fEVIon policy
P = 0
T = 1000
fEVIon_policy = fEVIDiscreteOnOff(n, m, theta0, Sigma0, sample_std, FX, P, T, labeling0)

## Biased Coin
p1 = 0.5
pk = vcat([p1], ones(n - 1) * (1 - p1) / (n - 1))
Gweight = [0, 0.5, 0.5, 0]
biasedcoin_policy = BiasedCoinPolicyLinear(n, m, theta0, Sigma0, sample_std, FX, labeling0;
    p=pk, weights=Gweight)

## OCBA
ocba_policy = OCBAPolicyLinear(n, m, theta0, Sigma0, sample_std, FX, labeling0)

## RABC
Gweight = [0.25, 0.25, 0.25, 0.25]
RABC_policy = RABC_OCBA_PolicyLinear(n, m, theta0, Sigma0, sample_std, FX, labeling0; p=pk, weights=Gweight)

## Greedy
greedy_policy = GreedyPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling0)

# policies
policies = Dict(
    "random" => random_policy, "TS" => TS_policy,
    "TTTS" => TTTS_policy, "fEVI" => fEVI_policy, "fEVIon" => fEVIon_policy,
    "biasedcoin" => biasedcoin_policy, "RABC" => RABC_policy, "ocba" => ocba_policy,
    "greedy" => greedy_policy
)

# Settings for simulation runs
T = 1000
reps = 5000 # number of replications
post_reps = 50
Xinterest = [
    1 1 1 1;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1;
    0 0 0 0;
    0 0 0 0
]

# run simulation
rng = Xoshiro(121)
results = @time simulation_stochastic_parallel(reps, FX, n, T, policies, outcome_model;
    FXtilde=FXtilde, delay=delay, post_reps=post_reps, rng=rng, Xinterest=Xinterest)


## Save
save("example.jld2", results)
```

### Plotting simulation data
Once the simulation is finished and the data has been saved, you can reload the data and plot it:

```julia
using JLD2
using Plots

# load data
output = load("example.jld2", "output")

# policies that will be plotted
policy_keys = ["fEVI", "TTTS", "RABC", "TS", "random", "fEVIon"]

# performance metric to plot (regret, cumulregret, pics)
metric = "regret_off"

# horizon T and sample size
T = length(output[policy_keys[1]]["regret_on"])

# data to plot
if occursin("off", metric) # offline metric
    x = 0:T
else # online metric
    x = 1:T
end
y = Matrix{Float64}(undef, length(x), length(policy_keys))
for i in eachindex(policy_keys)
    y[:, i] = output[policy_keys[i]][metric]["mean"]
end

plot(x, y, label=policy_keys)
```
