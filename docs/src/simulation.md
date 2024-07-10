Simulation
==========

```@docs
simulation_stochastic
```

The return value is a dictionary with two keys: `"input"` and `"output"`. The `"input"` saves the input parameters to the simulation. The `"output"` contains one key for each policy. For each policy, we output the following metrics:

1. `"regret_on"`: online regret
2. `"cumulregret_on"`: cumulative online regret
3. `"PICS_on"`: online PICS
4. `"cumulPICS_on"`: cumulative online PICS 
5. `"Wfrac_on"`: fraction of allocations to each treatment during trial
6. `"regret_off"`: offline regret
7. `"PICS_off"`: offline PICS 
8. `"Wfrac_off"`: fraction of allocations to each treatment after implementation

Online metrics are recorded for each time `t=1:T`. All offline metrics are recorded for each time `t=0:T`. Offline metrics at time`t` represent the metric in the scenario where the trial would have concluded at time `t`. For the eight online and offline metrics, the output also provides those metrics for each of the specific covariate values specified in `Xinterest`, which are saved under the same names with an `X` at the beginning, for example, `"Xregret_off"` measures the offline regret for subjects that have the covariate values specified in `Xinterest`.

Two additional outputs are provided that are useful for understanding policies that infer the labeling:

9. `"labeling_frac"`: Which entries of labeling where `true` when the trial concluded, which after aggregation represents the fraction of replications in which the entries labeling were `true`.
10. `"sum_labeling"`: The number of `true` entries of labeling.

## Parallel Computing

Because simulations can be computationally expensive, the `simulation_stochastic_parallel` is used to distribute the replications evenly among all available workers (see the standard library [Distributed Computing](https://docs.julialang.org/en/v1/stdlib/Distributed/) page on how to create worker processes).

```@docs
simulation_stochastic_parallel
```

An effective workflow is to run parallel simulations in a script. At the top of the script, you need to load the package in all workers with `@everywhere`:

```julia
# example_parallel.jl
using Distributed
@everywhere using ContextualBandits
# ...
# generate all the input arguments for the function
# ...
results = simulation_stochastic_parallel(FX, n, T, policies, outcome_model)
```
Then you run the script with the `-p` flag indicating the number of workers you want to use or `auto` to use all available cores ([command line interface for Julia](https://docs.julialang.org/en/v1/manual/command-line-interface/)):
```
julia -p auto example_parallel.jl
```
If you installed `ContextualBandits` in a local environment, you also need to activate the environment in all workers using `Pkg`:
```julia
# example_parallel.jl
using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(".") # path to the environment
@everywhere using ContextualBandits
# ...
# generate all the input arguments for the function
# ...
results = simulation_stochastic_parallel(FX, n, T, policies, outcome_model)
```
