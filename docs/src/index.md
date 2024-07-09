ContextualBandits.jl
====================

ContextualBandits.jl is a Julia package that implements contextual bandit policies and functionality to estimate regret and other metrics through simulation.
It was originally developed to simulate the contextual bandit policies in the paper [Alban A, Chick SE, Zoumpulis SI (2024) Learning Personalized Treatment Strategies with Predictive and Prognostic Covariates in Adaptive Clinical Trials](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4160045). Although the functionality of the package is broader than that of the paper, ContextualBandits.jl is heavily influenced by the paper's model, which focuses on rewards/signals that are linear function of the contextual information and policies that learn a Bayesian linear regression model.