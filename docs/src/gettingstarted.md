Getting Started
===============

## Installation

ContextualBandits can be installed using the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
pkg> add https://github.com/andres-alban/ContextualBandits.jl.git
```

!!! note
    The package is not in the registry but is installed directly from the Github repository.

Most often, you will also need the following packages:
```
pkg> add Distributions, LinearAlgebra
```

## Basic Usage

Below is an example of a simulation. [`simulation_stochastic`](@ref) is the main function that guides the development of policies and other components. We will next explain what `simulation_stochastic` does. Then, we explain what the inputs represent and how you can modify them for your purposes.

```julia
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
# vector of length T of cumulative regret
output_random["cumulregret_on"]
```
