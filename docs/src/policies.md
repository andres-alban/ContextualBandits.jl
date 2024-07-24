Policies
========

In [Model](@ref), we introduced the [`Policy`](@ref) abstract type. Here, we first introduced [`PolicyLinear`](@ref) that serves as the supertype for the policies with a linear model. Then, we introduce the concrete policies that are defined by the package. More detail about the policies defined here is provided in [Alban, Chick, Zoumpoulis (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4160045).

## Linear Policies

```@docs
PolicyLinear
```

### Expected Value of Information (EVI)

```@docs
fEVI_MC_PolicyLinear
```

```@docs
fEVIDiscrete
fEVIDiscreteOnOff
```
### Thompson sampling

```@docs
TSPolicyLinear
TTTSPolicyLinear
```

### Optimal Computing Budget Allocation (OCBA)
```@docs
OCBAPolicyLinear
```

### Biased coin

```@docs
BiasedCoinPolicyLinear
RABC_OCBA_PolicyLinear
```

## Policy Modifiers

### Infer labeling

```@docs
InferLabelingPolicy
```

### Discretize policy
```@docs
DiscretizePolicy
```