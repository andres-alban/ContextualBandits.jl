"""
    treatmentFromIndex(i,m)

Return the treatment from the index (position) `i` and the number of covariates `m`.
"""
function treatmentFromIndex(i,m)
    return (i-1) ÷ m
end

"""
    covariateFromIndex(i,m)

Return the covariate from the index (position) `i` and the number of covariates `m`.
"""
function covariateFromIndex(i,m)
    return ((i-1) % m) + 1
end

"""
    interact(w,Wn,x,labeling)

Interact a treatment index `w` among `Wn` treatment alternatives with covariates `x`, where the coefficients labeled by `labeling` are active.

The output is a matrix with `sum(labeling)` rows and `length(w)` columns.

See also: [`interact!`](@ref)

#Examples
```jldoctest
julia> w = [1,2];
       Wn = 2;
       x = [1 1;
            2 3;
            5 4];
       labeling = Bool.([1,1,0,0,0,1,0,1,0]);
       interact(w,Wn,x,labeling)
4×2 Matrix{Float64}:
 1.0  1.0
 2.0  3.0
 5.0  0.0
 0.0  3.0
```
"""
function interact(w,Wn,x,labeling=vcat(falses(size(x,1)),trues(Wn*size(x,1))))
    interact!(Matrix{Float64}(undef,sum(labeling),length(w)),w,Wn,x,labeling)
end

function interact(w::Integer,Wn,x,labeling=vcat(falses(size(x,1)),trues(Wn*size(x,1))))
    interact!(Vector{Float64}(undef,sum(labeling)),w,Wn,x,labeling)
end


"""
    interact!(WX,w,Wn,x,labeling=vcat(falses(size(x,2)),trues(Wn*size(x,2))))

In-place version of [`interact`](@ref).
"""
function interact!(WX,w,Wn,x,labeling=vcat(falses(size(x,1)),trues(Wn*size(x,1))))
    length(w) == size(x,2) || throw(DomainError(w,"`w` must be the same length as the number of columns of `x`."))
    m = size(x,1)
    d = sum(labeling)
    length(labeling) == (Wn+1)*m || throw(DomainError(labeling,"`labeling` must have length `(Wn+1)*m`, where `m=size(x,1)`."))
    size(WX,1) == d || throw(DomainError(WX,"`WX` must have `d` rows, where `d=sum(labeling)`."))
    size(WX,2) == length(w) || throw(DomainError(WX,"`WX` must have the same number of columns as `x`."))

    for i in axes(WX,2)
        index = 0
        for j in eachindex(labeling)
            if labeling[j]
                index += 1
                treat_index = treatmentFromIndex(j,m)
                cov_index = covariateFromIndex(j,m)
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




"""
    BayesUpdateNormal(theta,Sigma,X,y,sample_std)

Update Bayesian hyperparameters `theta` (mean vector) and `Sigma` (covariance matrix) of a linear regression model
with regressor matrix `X`, outputs `y`, and sampling standard deviation `sample_std`.

Return a copy of the updated `theta` and `Sigma`.

`X` can be a vector of regressors or a matrix of regressors, where each column is a vector of regressors.
In the latter case, `y` and `sample_std` can be vectors of the same length as the number of columns of `X`.

See also: [BayesUpdateNormal!](@ref)
"""
function BayesUpdateNormal(theta,Sigma,X,y,sample_std)
    BayesUpdateNormal!(copy(theta),copy(Sigma),X,y,sample_std)
end

"""
    BayesUpdateNormal!(theta,Sigma,X,y,sample_std)

In-place version of [BayesUpdateNormal](@ref)
"""
function BayesUpdateNormal!(theta,Sigma,X,y,sample_std)
    theta .= theta .+ (y .- X'*theta)./(sample_std^2 .+ X'*Sigma*X) .* Sigma*X
    Sigma .= Sigma - ((X' * Sigma)' * (X' *Sigma)) ./ (sample_std^2 .+ X'*Sigma*X)
    return theta,Sigma
end

function BayesUpdateNormal!(theta,Sigma,X,y::AbstractVector,sample_std)
    for i in eachindex(y)
        BayesUpdateNormal!(theta,Sigma,view(X,:,i),y[i],sample_std)
    end
    return theta,Sigma
end

function BayesUpdateNormal!(theta,Sigma,X,y::AbstractVector,sample_std::AbstractVector)
    for i in eachindex(y)
        BayesUpdateNormal!(theta,Sigma,view(X,:,i),y[i],sample_std[i])
    end
    return theta,Sigma
end


"""
    argmax_ties(itr,rng=Random.GLOBAL_RNG)

Select the argmax of `itr` solving ties uniformly at random.
"""
function argmax_ties(itr,rng=Random.GLOBAL_RNG)
    maxs = findall(itr .== maximum(itr))
    if length(maxs) == 1
        return maxs[1]
    else
        return rand(rng,maxs)
    end
end

"""
    argmin_ties(itr,rng=Random.GLOBAL_RNG)

Select the argmin of `itr` solving ties uniformly at random.
"""
function argmin_ties(itr,rng=Random.GLOBAL_RNG)
    mins = findall(itr .== minimum(itr))
    if length(mins) == 1
        return mins[1]
    else
        return rand(rng,mins)
    end
end

"""
    randnMv(rng, mu, Sigma)

Sample from a multivariate normal distribution with mean `mu` and covariance `Sigma`.
> Warning: this function does not check if `Sigma` is positive semidefinite.
"""
function randnMv(rng, mu, Sigma)
    # The following is a sample draw from a multivariate normal distribution that allows for a positive semidefinite covariance matrix.
    # The MvNormal distribution in Distributions.jl requires a positive definite covariance matrix.
    @static if VERSION >= v"1.8"
        chol = cholesky(Sigma,RowMaximum(),check=false)
    else
        chol = cholesky(Sigma,Val(true),check=false)
    end
    return mu .+ chol.L[invperm(chol.p),1:chol.rank]*randn(rng,chol.rank)
end