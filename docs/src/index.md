Introduction
============

ContextualBandits.jl is a Julia package that implements contextual bandit policies and functionality to estimate regret and other metrics through simulation.
It was originally developed to simulate the contextual bandit policies in the paper [Alban A, Chick SE, Zoumpoulis SI (2024) Learning Personalized Treatment Strategies with Predictive and Prognostic Covariates in Adaptive Clinical Trials](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4160045). Although the functionality of the package is broader than that of the paper, ContextualBandits.jl is heavily influenced by the paper's model, which focuses on rewards/signals that are linear function of the contextual information plus some noise and policies that learn a Bayesian linear regression model.

The [Getting Started](@ref) page shows how to install the package and the easiest way to start using it. However, we recommend that you read the [Model](@ref) page to understand some important concepts. Additional pages complement the [Model](@ref) page.

### If you are new to Julia

To install Julia follow the instructions at [https://julialang.org/downloads](https://julialang.org/downloads). The [Julia manual](https://docs.julialang.org/en/v1/manual/getting-started/) is a valuable resource, but you can start with one of the tutorials listed at [https://julialang.org/learning/tutorials/](https://julialang.org/learning/tutorials/), such as [From Zero to Julia](https://techytok.com/from-zero-to-julia/). Understanding the [package manager](https://pkgdocs.julialang.org/v1/getting-started/), which comes with the Julia installation, will help you understand how easy it is to install this package for your own use.