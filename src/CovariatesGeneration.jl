module CovariatesGeneration
using Distributions
using Copulas
using Random
import Base

export CovariatesGenerator, CovariatesGeneratorFinite
export CovariatesCopulaAny, CovariatesCopulaFinite, CovariatesCopula
export CovariatesIndependentAny, CovariatesIndependentFinite, CovariatesIndependent
export CovariatesInteracted, marginals, covariates_partition
export OrdinalDiscrete

"""
    CovariatesGenerator <: Sampleable{Multivariate,Continuous}

Supertype for CovariatesCopula and CovariatesIndependent.
"""
abstract type CovariatesGenerator <: Sampleable{Multivariate,Continuous} end

"""
    CovariatesCopula{C} <: CovariatesGenerator
    CovariatesCopula(marginals, copula, intercept=true, reduce_category=true)

Supertype for [CovariatesCopulaAny](@ref) and [CovariatesCopulaFinite](@ref). The constructor
generates one of these types based on the types of `marginals`. 
> Warning
> `CovariatesCopulaFinite` is generated if all marginals are discrete and univariate
> and will error if any marginal has an infinite support. In that case, use the
> `CovariatesCopulaAny` constructor instead, which takes the same arguments.

The two types are sampleables to randomly generate a vector of covariates using
marginal distributions and copulas from the Copulas.jl package.

`Categorical` marginal distributions are assumed unordered (nominal) and automatically generated as a set of dummy variables.
To use ordered categorical variables, use the [OrdinalDiscrete](@ref) type.

# Arguments
- `marginals::Vector{Distribution{Univariate,S} where S<:ValueSupport}`: vector of univariate distributions from the Distributions package
- `copula`: copula type from the `Copulas` package
- `intercept=true`: include an intercept term in the vector of covariates
- `reduce_category=true`: omit the first dummy variable for `Categorical` covariates

# Examples
```julia
using Distributions, Copulas
copula = GaussianCopula([1 0.2; 0.2 1])
FX = CovariatesCopula([Categorical([1/3,1/3,1/3]),Normal(0,1)],copula)
rand(FX)
```
"""
abstract type CovariatesCopula <: CovariatesGenerator end

"""
    CovariatesCopulaAny{C} <: CovariatesCopula

Sampleables to randomly generate a vector of covariates using
marginal distributions and copulas from the Copulas.jl package.

This type should be generated using the [CovariatesCopula](@ref) constructor.

`C` is a copula type from the `Copulas` package.
"""
struct CovariatesCopulaAny{C} <: CovariatesCopula
    marginals::Vector{Distribution{Univariate,S} where S<:ValueSupport}
    copula::C
    len::Int
    cat::BitVector
    partition::Vector{UnitRange{Int}}
    intercept::Bool
    reduce_category::Bool
    function CovariatesCopulaAny{C}(marginals, copula::C, intercept=true, reduce_category=true) where C
        length(copula) == length(marginals) || throw(DomainError(copula, "The copula has to have the same length as the marginals"))
        if intercept && reduce_category
            reduce_category = true
        else
            reduce_category = false
        end
        cat = BitVector(zeros(Bool, length(marginals)))
        partition = Vector{UnitRange{Int}}(undef, length(marginals))
        index = 1
        if intercept
            index += 1
        end
        for i in 1:length(marginals)
            if typeof(marginals[i]) <: Categorical
                cat[i] = true
                newindex = index + length(support(marginals[i])) - reduce_category
                partition[i] = index:newindex-1
                index = newindex
            else
                partition[i] = index:index
                index += 1
            end
        end
        new{C}(marginals, copula, index - 1, cat, partition, intercept, reduce_category)
    end
end
CovariatesCopulaAny(marginals, copula::C, intercept=true, reduce_category=true) where C =
    CovariatesCopulaAny{C}(marginals, copula::C, intercept, reduce_category)

function CovariatesCopulaAny(marginals, corr::Float64, intercept=true, reduce_category=true)
    corr_mat = fill(corr, (length(marginals), length(marginals)))
    for i in 1:length(marginals)
        corr_mat[i, i] = 1.0
    end
    copula = GaussianCopula(corr_mat)
    CovariatesCopulaAny(marginals, copula, intercept, reduce_category)
end

function CovariatesCopulaAny(marginals, corr_mat::AbstractMatrix{Float64}, intercept=true, reduce_category=true)
    copula = GaussianCopula(corr_mat)
    CovariatesCopulaAny(marginals, copula, intercept, reduce_category)
end

"""
    CovariatesCopulaFinite{C} <: CovariatesCopula

Sampleables to randomly generate a vector of covariates using
marginal distributions and copulas from the Copulas.jl package.

This type should be generated using the [CovariatesCopula](@ref) constructor.
This type is only generated if all marginals have a finite support.

`C` is a copula type from the `Copulas` package.
"""
struct CovariatesCopulaFinite{C} <: CovariatesCopula
    marginals::Vector{DiscreteUnivariateDistribution}
    copula::C
    len::Int
    cat::BitVector
    partition::Vector{UnitRange{Int}}
    support::Vector{Vector{Float64}}
    probs::Vector{Vector{Float64}}
    intercept::Bool
    reduce_category::Bool
    function CovariatesCopulaFinite{C}(marginals, copula::C, intercept=true, reduce_category=true) where C
        length(copula) == length(marginals) || throw(DomainError(copula, "The copula has to have the same length as the marginals"))
        if intercept && reduce_category
            reduce_category = true
        else
            reduce_category = false
        end
        cat = BitVector(zeros(Bool, length(marginals)))
        partition = Vector{UnitRange{Int}}(undef, length(marginals))
        index = 1
        if intercept
            index += 1
        end
        supp = Vector{Vector{Float64}}(undef, length(marginals))
        p = similar(supp)
        for i in 1:length(marginals)
            try
                supp[i] = collect(support(marginals[i]))
            catch
                throw(DomainError(marginals[i],
                    "CovariatesCopulaFinite can only be created with marginals that have finite support. Use CovariatesCopulaAny instead."))
            end
            p[i] = probs(marginals[i])
            if typeof(marginals[i]) <: Categorical
                cat[i] = true
                newindex = index + length(support(marginals[i])) - reduce_category
                partition[i] = index:newindex-1
                index = newindex
            else
                partition[i] = index:index
                index += 1
            end
        end
        new{C}(marginals, copula, index - 1, cat, partition, supp, p, intercept, reduce_category)
    end
end
CovariatesCopulaFinite(marginals, copula::C, intercept=true, reduce_category=true) where C =
    CovariatesCopulaFinite{C}(marginals, copula::C, intercept, reduce_category)

function CovariatesCopulaFinite(marginals::AbstractVector{<:Distribution{Univariate,S} where {S<:ValueSupport}}, corr::Float64, intercept=true, reduce_category=true)
    corr_mat = fill(corr, (length(marginals), length(marginals)))
    for i in 1:length(marginals)
        corr_mat[i, i] = 1.0
    end
    copula = GaussianCopula(corr_mat)
    CovariatesCopulaFinite(marginals, copula, intercept, reduce_category)
end

function CovariatesCopulaFinite(marginals::AbstractVector{<:Distribution{Univariate,S} where {S<:ValueSupport}}, corr_mat::AbstractMatrix{Float64}, intercept=true, reduce_category=true)
    copula = GaussianCopula(corr_mat)
    CovariatesCopulaFinite(marginals, copula, intercept, reduce_category)
end

# copula here can also be a correlation value/matrix
function CovariatesCopula(marginals::AbstractVector{<:DiscreteUnivariateDistribution}, copula, intercept=true, reduce_category=true)
    CovariatesCopulaFinite(marginals, copula, intercept, reduce_category)
end

function CovariatesCopula(marginals, copula, intercept=true, reduce_category=true)
    CovariatesCopulaAny(marginals, copula, intercept, reduce_category)
end

function Base.length(d::CovariatesCopula)::Int
    d.len
end

function marginals(d::CovariatesCopula)
    return d.marginals
end

function Distributions._rand!(rng::AbstractRNG, d::CovariatesCopula, x::AbstractVector{T}) where {T<:Real}
    cop = rand(rng, d.copula)
    x .= 0.0
    j = 1
    if d.intercept
        x[j] = 1
        j += 1
    end
    for i in 1:length(d.marginals)
        if d.cat[i]
            tmp = quantile(d.marginals[i], cop[i])
            if d.reduce_category
                if tmp > 1
                    x[j+tmp-2] = 1.0
                end
                j += length(d.marginals[i].p) - 1
            else
                x[j+tmp-1] = 1.0
                j += length(d.marginals[i].p)
            end
        else
            x[j] = quantile(d.marginals[i], cop[i])
            j += 1
        end
    end
    return x
end

"""
    CovariatesIndependent <: CovariatesGenerator
    CovariatesIndependent(marginals, intercept=true, reduce_category=true)

Supertype for [CovariatesIndependentAny](@ref) and [CovariatesIndependentFinite](@ref). The constructor
generates one of these types based on the types of `marginals`.
> Warning
> `CovariatesIndependentFinite` is generated if all marginals are discrete and univariate
> and will error if any marginal has an infinite support. In that case, use the
> `CovariatesCopulaAny` constructor instead, which takes the same arguments.

The two types are sampleables to randomly generate a vector of covariates using
independent marginal distributions.

`Categorical` marginal distributions are assumed unordered (nominal) and automatically generated as a set of dummy variables.
To use ordered categorical variables, use the [OrdinalDiscrete](@ref) type.

# Arguments
- `marginals::Vector{Distribution{Univariate,S} where S<:ValueSupport}`: vector of univariate distributions from the Distributions package
- `intercept=true`: include an intercept term in the vector of covariates
- `reduce_category=true`: omit the first dummy variable for `Categorical` covariates

# Examples
```julia
using Distributions
FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),Normal(0,1)])
rand(FX)
```
"""
abstract type CovariatesIndependent <: CovariatesGenerator end

"""
    CovariatesIndependentAny <: CovariatesIndependent

Sampleables to randomly generate a vector of covariates using
independent marginal distributions.

This type should be generated using the [CovariatesIndependent](@ref) constructor.
"""
struct CovariatesIndependentAny <: CovariatesIndependent
    marginals::Vector{Distribution{Univariate,S} where S<:ValueSupport}
    len::Int
    cat::BitVector
    partition::Vector{UnitRange{Int}}
    intercept::Bool
    reduce_category::Bool
    function CovariatesIndependentAny(marginals, intercept=true, reduce_category=true)
        if intercept && reduce_category
            reduce_category = true
        else
            reduce_category = false
        end
        cat = BitVector(zeros(Bool, length(marginals)))
        partition = Vector{UnitRange{Int}}(undef, length(marginals))
        index = 1
        if intercept
            index += 1
        end
        for i in 1:length(marginals)
            if typeof(marginals[i]) <: Categorical
                cat[i] = true
                newindex = index + length(support(marginals[i])) - reduce_category
                partition[i] = index:newindex-1
                index = newindex
            else
                partition[i] = index:index
                index += 1
            end
        end
        new(marginals, index - 1, cat, partition, intercept, reduce_category)
    end
end

"""
    CovariatesIndependentFinite <: CovariatesIndependent

Sampleables to randomly generate a vector of covariates using
independent marginal distributions.

This type should be generated using the [CovariatesIndependent](@ref) constructor.
This type is only generated if all marginals have a finite support.
"""
struct CovariatesIndependentFinite <: CovariatesIndependent
    marginals::Vector{DiscreteUnivariateDistribution}
    len::Int
    cat::BitVector
    partition::Vector{UnitRange{Int}}
    support::Vector{Vector{Float64}}
    probs::Vector{Vector{Float64}}
    intercept::Bool
    reduce_category::Bool
    function CovariatesIndependentFinite(marginals, intercept=true, reduce_category=true)
        if intercept && reduce_category
            reduce_category = true
        else
            reduce_category = false
        end
        cat = BitVector(zeros(Bool, length(marginals)))
        partition = Vector{UnitRange{Int}}(undef, length(marginals))
        index = 1
        if intercept
            index += 1
        end
        supp = Vector{Vector{Float64}}(undef, length(marginals))
        p = similar(supp)
        for i in 1:length(marginals)
            try
                supp[i] = collect(support(marginals[i]))
            catch
                throw(DomainError(marginals[i],
                    "CovariatesIndependentFinite can only be created with marginals that have finite support. Use CovariatesIndependentAny instead."))
            end
            p[i] = probs(marginals[i])
            if typeof(marginals[i]) <: Categorical
                cat[i] = true
                newindex = index + length(support(marginals[i])) - reduce_category
                partition[i] = index:newindex-1
                index = newindex
            else
                partition[i] = index:index
                index += 1
            end
        end
        new(marginals, index - 1, cat, partition, supp, p, intercept, reduce_category)
    end
end

function CovariatesIndependent(marginals::AbstractVector{S}, intercept=true, reduce_category=true) where S<:DiscreteUnivariateDistribution
    CovariatesIndependentFinite(marginals, intercept, reduce_category)
end

function CovariatesIndependent(marginals, intercept=true, reduce_category=true)
    CovariatesIndependentAny(marginals, intercept, reduce_category)
end

function Base.length(d::CovariatesIndependent)::Int
    d.len
end

function marginals(d::CovariatesIndependent)
    return d.marginals
end

function Distributions._rand!(rng::AbstractRNG, d::CovariatesIndependent, x::AbstractVector{T}) where {T<:Real}
    x .= 0.0
    j = 1
    if d.intercept
        x[j] = 1
        j += 1
    end
    for i in 1:length(d.marginals)
        if d.cat[i]
            tmp = rand(rng, d.marginals[i])
            if d.reduce_category
                if tmp > 1
                    x[j+tmp-2] = 1.0
                end
                j += length(d.marginals[i].p) - 1
            else
                x[j+tmp-1] = 1.0
                j += length(d.marginals[i].p)
            end
        else
            x[j] = rand(rng, d.marginals[i])
            j += 1
        end
    end
    return x
end

const CovariatesGeneratorFinite = Union{CovariatesCopulaFinite,CovariatesIndependentFinite}

"""
    covariates_partition(FX::Union{CovariatesIndependent,CovariatesCopula})

Partition the covariates into indices representing each covariate.
Categorical covariates are represented by several indices, while others by a single index.
"""
function covariates_partition(FX::Union{CovariatesIndependent,CovariatesCopula})
    return FX.partition
end

"""
    CovariatesInteracted{T<:Sampleable} <: Sampleable{Multivariate,Continuous}
    CovariatesInteracted(generator::T,interact_functions::Vector{Function})

Sampleable to randomly generate a vector of covariates using a base generator and a vector of functions to interact the covariates from the base generator.

# Examples
```julia 
using Distributions
FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),Normal(0,1)])
f(x) = x[1]*x[2]
FXinteracted = CovariatesInteracted(FX,[f])
rand(FXinteracted)
```
"""
struct CovariatesInteracted{T<:Sampleable} <: Sampleable{Multivariate,Continuous}
    generator::T
    interact_functions::Vector{Function}
end

Base.length(d::CovariatesInteracted) = length(d.interact_functions)

function marginals(d::CovariatesInteracted)
    return marginals(d.generator)
end

function Distributions._rand!(rng::AbstractRNG, d::CovariatesInteracted, x::AbstractVector{T}) where {T<:Real}
    X_base = rand(rng, d.generator)
    for i in eachindex(x)
        x[i] = d.interact_functions[i](X_base)
    end
    return x
end

# Most of the following code was adapted from the Distributions package at 
# https://github.com/JuliaStats/Distributions.jl/blob/c664d2591dd823483240b799387b049dc5f6851b/src/univariate/discrete/discretenonparametric.jl
"""
    struct OrdinalDiscrete{T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractVector{P}} <: DiscreteUnivariateDistribution

Type to sample from a discrete distribution. Unlike `DiscreteNonParametric` or `Categorical`,
this type is interpreted as ordinal by `CovariatesCopula` and `CovariatesIndependent`.
> Warning: although this type is a Distribution, I have only defined the rand and 
> quantile functions, which are required for covariate generation.

The implementation wraps a `DiscreteNonParametric` variable in a new structure.
"""
struct OrdinalDiscrete{T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractVector{P}} <: DiscreteUnivariateDistribution
    distribution::DiscreteNonParametric{T,P,Ts,Ps}

    function OrdinalDiscrete{T,P,Ts,Ps}(xs::Ts, ps::Ps; check_args::Bool=true) where {
        T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractVector{P}}
        new{T,P,Ts,Ps}(DiscreteNonParametric{T,P,Ts,Ps}(xs, ps; check_args=check_args))
    end
end

OrdinalDiscrete(vs::AbstractVector{T}, ps::AbstractVector{P}; check_args::Bool=true) where {T<:Real,P<:Real} =
    OrdinalDiscrete{T,P,typeof(vs),typeof(ps)}(vs, ps; check_args=check_args)

OrdinalDiscrete(ps::AbstractVector{P}; check_args::Bool=true) where {P<:Real} =
    OrdinalDiscrete(0:length(ps)-1, ps; check_args=check_args)

Base.eltype(::Type{<:OrdinalDiscrete{T}}) where {T} = T

Distributions.params(d::OrdinalDiscrete) = params(d.distribution)

Distributions.support(d::OrdinalDiscrete) = support(d.distribution)

Distributions.probs(d::OrdinalDiscrete) = probs(d.distribution)

function Base.rand(rng::AbstractRNG, d::OrdinalDiscrete)
    return rand(rng, d.distribution)
end

Distributions.sampler(d::OrdinalDiscrete) = DiscreteNonParametricSampler(support(d), probs(d))

function Distributions.quantile(d::OrdinalDiscrete, q::Real)
    return quantile(d.distribution, q)
end

end # module CovariatesGeneration