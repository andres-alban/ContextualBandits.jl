"""
    interact(w::AbstractVector{<:Integer},Wn::Integer,x::AbstractMatrix{<:Real},labels::AbstractVector{Bool})

Interact a treatment index `w` among `Wn` treatment alternatives with covariates `x`, where the coefficients labeled by `labels` are active.

The output is a matrix with `sum(labels)` rows and `length(w)` columns. If `labels` is not provided it is assumed to be a vector of all true.

See also: [`interact!`](@ref)

#Examples
```jldoctest
julia> w = [1,2];
       Wn = 2;
       x = [1 1;
            2 3;
            5 4];
       labels = Bool.([1,1,0,0,0,1,0,1,0]);
       interact(w,Wn,x,labels)
4×2 Matrix{Float64}:
 1.0  1.0
 2.0  3.0
 5.0  0.0
 0.0  3.0
```
"""
function interact(w::AbstractVector{<:Integer},Wn::Integer,x::AbstractMatrix,labels::AbstractVector{Bool})
    interact!(Matrix{Float64}(undef,sum(labels),length(w)),w,Wn,x,labels)
end


"""
    interact!(WX::AbstractMatrix{<:Real},w::AbstractVector{<:Integer},Wn::Integer,x::AbstractMatrix{<:Real},labels::AbstractVector{Bool})
    interact!(WX::AbstractMatrix{<:Real},w::Integer,Wn::Integer,x::AbstractMatrix{<:Real},labels::AbstractVector{Bool})

In-place version of [`interact`](@ref).
"""
function interact!(WX::AbstractMatrix,w::AbstractVector{<:Integer},Wn::Integer,x::AbstractMatrix,labels::AbstractVector{Bool})
    length(w) == size(x,2) || throw(DomainError(w,"`w` must be the same length as the number of columns of `x`."))
    m = size(x,1)
    d = sum(labels)
    length(labels) == (Wn+1)*m || throw(DomainError(labels,"`labels` must have length `(Wn+1)*m`, where `m=size(x,1)`."))
    size(WX,1) == d || throw(DomainError(WX,"`WX` must have `d` rows, where `d=sum(labels)`."))
    size(WX,2) == length(w) || throw(DomainError(WX,"`WX` must have the same number of columns as `x`."))

    for i in axes(WX,2)
        index = 0
        for j in eachindex(labels)
            if labels[j]
                index += 1
                treat_index = div(j-1,m)
                cov_index = ((j-1) % m) + 1
                if (treat_index == 0 || treat_index == w[i])
                    WX[index,i] = x[cov_index,i]
                else
                    WX[index,i] = 0
                end
            end
        end
    end

    return WX
end

# One-dimensional inputs 
function interact!(WX::AbstractVector,w::Integer,Wn::Integer,x::AbstractVector,labels::AbstractVector{Bool})
    m = length(x)
    d = sum(labels)
    length(labels) == (Wn+1)*m || throw(DomainError(labels,"`labels` must have length `(Wn+1)*m`, where `m=size(x,1)`."))
    length(WX) == d || throw(DomainError(WX,"`WX` must have `d` rows, where `d=sum(labels)`."))

    index = 0
    for j in eachindex(labels)
        if labels[j]
            index += 1
            treat_index = div(j-1,m)
            cov_index = ((j-1) % m) + 1
            if (treat_index == 0 || treat_index == w)
                WX[index] = x[cov_index]
            else
                WX[index] = 0
            end
        end
    end
    return WX
end

"""
    interact(w::Integer,Wn::Integer,x::AbstractVector{<:Real},labels::AbstractVector{Bool})

When `w` is an integer and `x` a vector, the output is a vector of length `sum(labels)`.

#Examples
```jldoctest
julia> interact(1,2,[1,2,5],Bool.([1,1,0,0,0,1,0,1,0]))
4-element Vector{Float64}:
 1.0
 2.0
 5.0
 0.0
```
"""
function interact(w::Integer,Wn::Integer,x::AbstractVector,labels::AbstractVector{Bool})
    interact!(Vector{Float64}(undef,sum(labels)),w,Wn,x,labels)
end

# Default labels = ones(Bool,(Wn+1)*m)
function interact!(WX,w,Wn,x)
    interact!(WX,w,Wn,x,ones(Bool,(Wn+1)*size(x,1)))
end

function interact(w,Wn,x)
    interact(w,Wn,x,ones(Bool,(Wn+1)*size(x,1)))
end





"""
    BayesUpdateNormal(theta,Sigma,X,y,sample_std)

Update Bayesian hyperparameters `theta` (mean vector) and `Sigma` (covariance matrix) of a linear regression model
with regressor matrix `X`, outputs `y`, and sampling standard deviation `sample_std`.

Return a copy of the updated `theta` and `Sigma`.

`X` can be a vector of regressors or a matrix of regressors, where each column is a vector of regressors.
In the latter case, `y` and `sample_std` can be vectors of the same length as the number of columns of `X`.

Moreover, `X` can be an integer (or a vector of integers), which means that the vector of regressors is all zeros except for the `X`-th element, which is a one. 

See also: [BayesUpdateNormal!](@ref)
"""
function BayesUpdateNormal(theta,Sigma,X,y,sample_std)
    BayesUpdateNormal!(copy(theta),copy(Sigma),X,y,sample_std)
end

"""
    BayesUpdateNormal!(theta,Sigma,X,y,sample_std)

In-place version of [BayesUpdateNormal](@ref)
"""
function BayesUpdateNormal!(theta,Sigma,X::AbstractVector,y,sample_std)
    theta .= theta .+ (y .- X'*theta)./(sample_std^2 .+ X'*Sigma*X) .* Sigma*X
    Sigma .= Sigma - ((X' * Sigma)' * (X' *Sigma)) ./ (sample_std^2 .+ X'*Sigma*X)
    return theta,Sigma
end

function BayesUpdateNormal!(theta,Sigma,X::AbstractMatrix,y::AbstractVector,sample_std)
    for i in eachindex(y)
        BayesUpdateNormal!(theta,Sigma,view(X,:,i),y[i],sample_std)
    end
    return theta,Sigma
end

function BayesUpdateNormal!(theta,Sigma,X::AbstractMatrix,y::AbstractVector,sample_std::AbstractVector)
    for i in eachindex(y)
        BayesUpdateNormal!(theta,Sigma,view(X,:,i),y[i],sample_std[i])
    end
    return theta,Sigma
end

# When passing g instead of X, assume that the regressor vector is all zeros except for the g-th element, which is a one.
function BayesUpdateNormal!(theta,Sigma,g::Integer,y,sample_std)
    theta .= theta .+ (y .- theta[g])./(sample_std^2 .+ Sigma[g,g]) .* Sigma[:,g]
    Sigma .= Sigma - (Sigma[:,g] * Sigma[:,g]') ./ (sample_std^2 .+ Sigma[g,g])
    return theta,Sigma
end

function BayesUpdateNormal!(theta,Sigma,g::AbstractVector{<:Integer},y::AbstractVector,sample_std)
    for i in eachindex(y)
        BayesUpdateNormal!(theta,Sigma,g[i],y[i],sample_std)
    end
    return theta,Sigma
end

function BayesUpdateNormal!(theta,Sigma,g::AbstractVector{<:Integer},y::AbstractVector,sample_std::AbstractVector)
    for i in eachindex(y)
        BayesUpdateNormal!(theta,Sigma,g[i],y[i],sample_std[i])
    end
    return theta,Sigma
end

"""
    argmax_ties(itr,rng::AbstractRNG=Random.GLOBAL_RNG)

Select the argmax of `itr` solving ties uniformly at random.
"""
function argmax_ties(itr,rng::AbstractRNG=Random.GLOBAL_RNG)
    maxs = findall(itr .== maximum(itr))
    if length(maxs) == 1
        return maxs[1]
    else
        return rand(rng,maxs)
    end
end

"""
    argmin_ties(itr,rng::AbstractRNG=Random.GLOBAL_RNG)

Select the argmin of `itr` solving ties uniformly at random.
"""
function argmin_ties(itr,rng::AbstractRNG=Random.GLOBAL_RNG)
    mins = findall(itr .== minimum(itr))
    if length(mins) == 1
        return mins[1]
    else
        return rand(rng,mins)
    end
end