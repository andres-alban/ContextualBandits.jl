module CovariatesGeneration
using Distributions
using Copulas
using Random
import Base

export CovariatesCopula, CovariatesIndependent, CovariatesInteracted, OrdinalDiscrete, marginals, covariates_partition

"""
    CovariatesCopula{C} <: Sampleable{Multivariate,Continuous}
    CovariatesCopula(marginals,copula,intercept=true,reduce_category=true)

Sampleable to randomly generate a vector of covariates using marginal distributions and copulas from the Copulas.jl package.

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
struct CovariatesCopula{C} <: Sampleable{Multivariate,Continuous}
    marginals::Vector{Distribution{Univariate,S} where S<:ValueSupport}
    copula::C
    len::Int
    cat::BitVector
    intercept::Bool
    reduce_category::Bool
    function CovariatesCopula(marginals,copula,intercept=true,reduce_category=true)
        length(copula) == length(marginals) || throw(DomainError(copula,"The copula has to have the same length as the marginals"))
        if intercept && reduce_category
            reduce_category = true
        else
            reduce_category = false
        end
        len = 0
        cat = BitVector(zeros(Bool,length(marginals)))
        for i in 1:length(marginals)
            if typeof(marginals[i]) <: Categorical
                cat[i] = true
                len += length(support(marginals[i]))
                if reduce_category
                    len -=1
                end
            else
                len += 1
            end
        end
        if intercept
            len += 1
        end
        new{typeof(copula)}(marginals, copula, len, cat, intercept, reduce_category)
    end  
end

function CovariatesCopula(marginals::Array{<:Distribution{Univariate,S} where S<:ValueSupport,1},corr::Float64,intercept=true,reduce_category=true)
    corr_mat = fill(corr,(length(marginals),length(marginals)))
    for i in 1:length(marginals)
        corr_mat[i,i] = 1.
    end
    copula = GaussianCopula(corr_mat)
    CovariatesCopula(marginals, copula, intercept, reduce_category)
end

function CovariatesCopula(marginals::Array{<:Distribution{Univariate,S} where S<:ValueSupport,1},corr_mat::Array{Float64,2},intercept=true,reduce_category=true)
    copula = GaussianCopula(corr_mat)
    CovariatesCopula(marginals, copula, intercept, reduce_category)
end

function Base.length(d::CovariatesCopula)::Int
    d.len
end

function marginals(d::CovariatesCopula)
    return d.marginals
end

function Distributions._rand!(rng::AbstractRNG,d::CovariatesCopula, x::AbstractVector{T}) where T<:Real
    cop = rand(rng,d.copula)
    x .= 0.0
    j = 1
    if d.intercept
        x[j]=1
        j += 1
    end
    for i in 1:length(d.marginals)
        if d.cat[i]
            tmp = quantile(d.marginals[i],cop[i])
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
            x[j] = quantile(d.marginals[i],cop[i])
            j += 1  
        end
    end
    return x
end

"""
    CovariatesIndependent <: Sampleable{Multivariate,Continuous}
    CovariatesIndependent(marginals, intercept=true, reduce_category=true)

Sampleable to randomly generate a vector of covariates using marginal distributions assuming the values are independent of each other.

# Arguments
- `marginals::Array{Distribution{Univariate,S} where S<:ValueSupport,1}`: vector of univariate distributions from the Distributions package
- `intercept=true`: include an intercept term in the vector of covariates
- `reduce_category=true`: omit the first dummy variable for `Categorical` covariates

# Examples
```julia
using Distributions
FX = CovariatesIndependent([Categorical([1/3,1/3,1/3]),Normal(0,1)])
rand(FX)
```
"""
struct CovariatesIndependent <: Sampleable{Multivariate,Continuous}
    marginals::Vector{Distribution{Univariate,S} where S<:ValueSupport}
    len::Int
    cat::BitVector
    intercept::Bool
    reduce_category::Bool
    function CovariatesIndependent(marginals,intercept=true,reduce_category=true)
        if intercept && reduce_category
            reduce_category = true
        else
            reduce_category = false
        end
        len = 0
        cat = BitVector(zeros(Bool,length(marginals)))
        for i in 1:length(marginals)
            if typeof(marginals[i]) <: Categorical
                cat[i] = true
                len += length(support(marginals[i]))
                if reduce_category
                    len -=1
                end
            else
                len += 1
            end
        end
        if intercept
            len += 1
        end

        new(marginals, len, cat, intercept, reduce_category)
    end
end

function Base.length(d::CovariatesIndependent)::Int
    d.len
end

function marginals(d::CovariatesIndependent)
    return d.marginals
end

function Distributions._rand!(rng::AbstractRNG,d::CovariatesIndependent, x::AbstractVector{T}) where T<:Real
    x .= 0.0
    j = 1
    if d.intercept
        x[j]=1
        j += 1
    end
    for i in 1:length(d.marginals)
        if d.cat[i]
            tmp = rand(rng,d.marginals[i])
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
            x[j] = rand(rng,d.marginals[i])
            j += 1
        end
    end
    return x
end

"""
    covariates_partition(FX::Union{CovariatesCopula, CovariatesIndependent})

Partition the covariates into indices representing each covariate.
Categorical covariates are represented by several indices, while others by a single index.
"""
function covariates_partition(FX::Union{CovariatesCopula, CovariatesIndependent})
    partition = Vector{Vector{Int}}(undef, 0)
    index = 1
    if FX.intercept
        index += 1
    end
    for i in 1:length(FX.marginals)
        if FX.cat[i]
            len = length(support(FX.marginals[i]))-FX.reduce_category
            partition_candidate = collect(index:(index+len-1))
            if !isempty(partition_candidate)
                push!(partition, partition_candidate)
            end
            index += len
        else
            push!(partition, [index])
            index += 1
        end
    end
    return partition
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

function Distributions._rand!(rng::AbstractRNG,d::CovariatesInteracted, x::AbstractVector{T}) where T<:Real
    X_base = rand(rng,d.generator)
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
> WARNING: although this type is a Distribution, I have only defined the rand and 
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

Base.eltype(::Type{<:OrdinalDiscrete{T}}) where T = T

Distributions.params(d::OrdinalDiscrete) = params(d.distribution)

Distributions.support(d::OrdinalDiscrete) = support(d.distribution)

Distributions.probs(d::OrdinalDiscrete) = probs(d.distribution)

function Base.rand(rng::AbstractRNG, d::OrdinalDiscrete)
    return rand(rng, d.distribution)
end

Distributions.sampler(d::OrdinalDiscrete) = DiscreteNonParametricSampler(support(d), probs(d))

function Distributions.quantile(d::OrdinalDiscrete, q::Real)
    return quantile(d.distribution,q)
end

end # module CovariatesGeneration