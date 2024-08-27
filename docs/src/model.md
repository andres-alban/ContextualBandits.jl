Model
=====

There are many "flavors" of bandit models and this package is not comprehensive. Therefore, users must understand the underlying model for which this package was designed. The contextual bandit model that we present here was formally introduced in the paper [Alban A, Chick SE, Zoumpoulis SI (2024) Learning Personalized Treatment Strategies with Predictive and Prognostic Covariates in Adaptive Clinical Trials](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4160045). 
We introduce some terminology that will be used throughout this documentation. 

A **trial** is a sequence of allocations of `T` **subjects** to one of `n` alternative **treatments** (arms). Before allocating each subject, we observe a context, which is a vector with `m` **covariates**. After each allocation of a subject, we observe the **outcome**, a noisy scalar that represents how effective the treatment is for the subject (higher values are better). By the end of the trial, we have gathered the following data:
1. `W`: a `T`-dimensional vector of integers in `1:n` of the treatments allocated to each subject.
2. `X`: a `m`x`T`-dimensional matrix where each column is the vector of covariates for a subject. Notice that Julia stores arrays in [column-major order](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-column-major).
3. `Y`: a `T`-dimensional vector of scalars with the outcome of each subject.


### Policy
A **policy** makes allocation decisions: it decides to which treatment each subject is allocated. A policy is represented by a subtype of `Policy`:
```@docs
Policy
```
The main method for any subtype of policy is `allocation`:
```@docs
ContextualBandits.allocation(::Policy,Xcurrent,W,X,Y)
```

The allocation depends only on the available data `W,X,Y` when the allocation is made. However, the policy can also maintain a **state**. A state can be useful for computational efficiency. For instance, a linear regression model can be updated sequentially rather than retrained for every allocation. To maintain a state, the policy is first initialized and then updated after every observation:
```@docs
ContextualBandits.initialize!(::Policy)
ContextualBandits.state_update!(::Policy,W,X,Y)
```

A policy may also define an **implementation** that determines which treatment a subject arriving post-trial (once the trial has concluded and no more data is being gathered):

```@docs
ContextualBandits.implementation(::Policy,X_Post,W,X,Y)
```

In [Policies](@ref), we present the policies provided by this package. You can define your own policy by creating a subtype of `Policy` that defines methods for the above functions.

### Covariates
The covariates `X` are random draws from a distribution `FX`, which are independent between subjects. The package provides [`CovariatesIndependent`](@ref) and [`CovariatesCopula`](@ref), which are described in [Covariates Generation](@ref), to randomly generate covariates. As an example, a covariate that follows a standard normal distribution can be generated as follows:

```julia
using ContextualBandits
using Distributions

FX = CovariatesIndependent([Normal()])
X = rand(FX)
```

### Outcomes
The outcomes of a trial, `Y`, are generated by an **outcome model**:

```@docs
OutcomeModel
```

```@docs
ContextualBandits.mean_outcome
ContextualBandits.noise_outcome
ContextualBandits.noisy_outcome
ContextualBandits.outcome_model_state!
```

In [Outcome Models](@ref), we present the outcome models provided by this package. You can define your own model by creating a subtype of `OutcomeModel` that defines methods for the above functions.

### Delay

The model allows for fixed delays in observing the outcomes. [`simulation_stochastic`](@ref) accepts `delay` as an integer parameter that determines the number of subjects that are allocated before the outcome of a subject is observed. For example if `delay=3`, then the outcome of subject 1 will be observed only after the allocation of subject 4.

### Simulation of a Trial

The model can be understood with the following code snippet that broadly illustrates how the different components enter in a simulation. A simulation will perform this code for many replications, record metrics, and summarize the metrics. See [Simulation](@ref) for more details on the simulation functions.

```julia
# ...
# loop over subjects in trial
initialize!(policy)
for t in 0:(T+delay)
    # Allocate
    if 1 <= t <= T
        # Covariates of the newly arrived subject that has not yet been treated
        Xcurrent = rand(FX)
        w = allocation(policy,Xcurrent,view(W,1:(t-1)),view(X,:,1:(t-1)),view(Y,1:(t-delay-1)),rng)
        @assert w in 1:n "The policy $(typeof(policy)) made an allocation outside the range 1-$n"
        X[:,t] = Xcurrent
        W[t] = w
        Y[t] = noisy_outcome(outcome_model,w,view(X,:,t),Z[t])
    end

    # Available data at time t
    Wav = view(W,1:min(t,T))
    Xav = view(X,:,1:min(t,T))
    Yav = view(Y,1:(t-delay))

    # Update state of policy
    state_update!(policy,Wav,Xav,Yav,rng)
end
```

### Metrics and performance
We distinguish between two types of metrics: **online** and **offline**. Online performance metrics refer to performance associated with the subjects allocated in the trial, while offline refers to the outcome of subjects that would be allocated to treatment post-trial once an implementation has been chosen. **Regret** for a subject with covariates `x` allocated to treatment `w` is defined as follows:
```
regret = maximum([mean_outcome(outcome_model, wmax, x) - mean_outcome(outcome_model, w, x) for wmax in 1:n])
```
For subjects allocated in the trial, we refer to it as online regret, and for subjects that received the treatment as the post-trial implementation, we refer to it as offline regret. In [Alban, Chick, Zoumpoulis (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4160045) offline regret is referred to as Expected Opportunity Cost (EOC).

Another important metric is **Probability of Incorrect Selection (PICS)**:
```
PICS = regret > 0
```
!!! note
    This package assumes that larger values of the output are better when computing
    regret and for designing policies.

More detail is provided in the output of the [Simulation](@ref).

## Linear Model with Labeling
A common way to model the the outcomes as a function of the covariates and treatment is through linear model. Most effective policies require a statistical model that learns the coefficients of the linear model. The vast majority of implemented policies in this package use [Bayesian Linear Regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression) to learn those coefficients. We use `mu` as the true coefficients of the model that are generating the outcomes. `theta` is the prior/posterior estimate of `mu` and `Sigma` is the prior/posterior covariance matrix of the Bayesian linear regression.

A reasonable way to implement a linear model is to model each treatment separately. However, we use a more flexible framework of interaction between treatment and covariates with a `labeling` (see [Labeling](@ref) for details). Thus, the final model is `Y = interact(W,n,X,labeling) * mu + sigma * randn()`, where `sigma^2` is the standard deviation of the noise. [Alban, Chick, Zoumpoulis (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4160045) uses the operator ``\otimes`` to represent the `interact` function.
 
```@docs
interact
interact!
```

### Labeling

A labeling "labels" how each covariate interacts with each treatment. It is a boolean vector of size `(n+1)*m` split into `n+1` blocks of `m` entries. The first `m` entries represent **prognostic** covariates, those covariates that have an equal effect on all treatments. By default, all prognostic labels are `false`, meaning there are no prognostic effects. The remaining `n` blocks represent the **predictive** covariates with respect to the corresponding treatment, those covariates that have a treatment-specific effect. By default, all predictive labels are `true`, meaning each covariates has an effect specific to each treatment, or equivalently, each treatment has a separate model.

Generally, the first covariate is an intercept term that is `1` for all subjects. Thus, the first prognostic term represents an intercept term, and the first term for each block of predictive covariates are treatment effects.

### Bayesian linear regression
We use the following type to learn Bayesian linear regression models. Policies
will often use this type
```@docs
BayesLinearRegression
```

This type needs to be first initialized and then data can be passed sequentially (or in batches) to update the posterior distribution.
```@docs
ContextualBandits.initialize!(::BayesLinearRegression)
ContextualBandits.state_update!(::BayesLinearRegression,W,X,Y)
```

To select a prior, the package provides the following utilities to generate the robust prior of [Alban, Chick, Zoumpoulis (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4160045).

```@docs
default_prior_linear
robustify_prior_linear!
```

### Additional utilities
```@docs
BayesUpdateNormal
BayesUpdateNormal!
```