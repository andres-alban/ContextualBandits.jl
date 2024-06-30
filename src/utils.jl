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
    m = size(x,1)

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