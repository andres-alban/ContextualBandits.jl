"""
    treatmentFromIndex(i,m)

Return the treatment from the index (position) `i` and the number of covariates `m`.
"""
function treatmentFromIndex(i,m)
    return (i-1) ÷ m
end

function indexFromLabeling(i, labeling)
    for j in eachindex(labeling)
        if labeling[j]
            i -= 1
            if i == 0
                return j
            end
        end
    end
    return 0
end

function treatmentFromIndex(i,m, labeling)
    index = indexFromLabeling(i, labeling)
    return treatmentFromIndex(index,m)
end

"""
    covariateFromIndex(i,m)

Return the covariate from the index (position) `i` and the number of covariates `m`.
"""
function covariateFromIndex(i,m)
    return ((i-1) % m) + 1
end

function covariateFromIndex(i,m, labeling)
    index = indexFromLabeling(i, labeling)
    return covariateFromIndex(index,m)
end

"""
    indexFromTreatmentCovariate(w,j,m)

Return the index (position) from the treatment `w`, the covariate `j`, and the number of covariates `m`.
"""
function indexFromTreatmentCovariate(w,j,m)
    return w*m + j
end

function indexFromTreatmentCovariate(w,j,m,labeling)
    index = indexFromTreatmentCovariate(w,j,m)
    if labeling[index]
        return sum(labeling[1:index])
    else
        return 0
    end
end

"""
    interact(w, n, x[, labeling])

Interact a treatment `w` among `n` treatment alternatives with covariate vector `x` (`m=length(x)`).

When labeling is not provided, the function returns a vector `WX` of length `m*n`,
where all entries are zeros except for `WX[(w-1)*m+1:w*m] = x`. `WX` is split into
`n` blocks of size `m`, where the `w`-th block is equal to `x`.

When `labeling` is provided, the function returns a vector `WX` of length `sum(labeling)`.
`labeling` is a boolean vector of length `(n+1)*m` that indicates which covariates are predictive
and which are prognostic. The first `m` entries correspond to prognostic terms,
the following `m` entries correspond to the terms predictive with respect to treatment 1, then `m` entries for treatment 2, and so on.

If `w` is a vector, then `x` must be a matrix such that `length(w)==size(x,2)`,
and the output is a matrix with `sum(labeling)` rows and `length(w)` columns.

See also: [`interact!`](@ref)

#Examples
```jldoctest
julia> w = 2;
       n = 2;
       x = [1,3,4]
       interact(w, n, x)
Vector{Float64}:
 0.0
 0.0
 0.0
 1.0
 3.0
 4.0
```

```jldoctest
julia> w = [1,2]
       n = 2
       x = [1 1;
            2 3;
            5 4]
       labeling = Bool.([1,1,0,0,0,1,0,1,0])
       interact(w,n,x,labeling)
4×2 Matrix{Float64}:
 1.0  1.0
 2.0  3.0
 5.0  0.0
 0.0  3.0
```
"""
function interact(w,n,x,labeling=vcat(falses(size(x,1)),trues(n*size(x,1))))
    interact!(Matrix{Float64}(undef,sum(labeling),length(w)),w,n,x,labeling)
end

function interact(w::Integer,n,x,labeling=vcat(falses(size(x,1)),trues(n*size(x,1))))
    interact!(Vector{Float64}(undef,sum(labeling)),w,n,x,labeling)
end


"""
    interact!(WX, w, n, x[, labeling])

In-place version of [`interact`](@ref).
"""
function interact!(WX,w,n,x,labeling=vcat(falses(size(x,1)),trues(n*size(x,1))))
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
    argmax_ties(itr,rng=Random.default_rng())

Select the argmax of `itr` solving ties uniformly at random.
"""
function argmax_ties(itr,rng=Random.default_rng())
    maxs = findall(itr .== maximum(itr))
    if length(maxs) == 1
        return maxs[1]
    else
        return rand(rng,maxs)
    end
end

"""
    argmin_ties(itr,rng=Random.default_rng())

Select the argmin of `itr` solving ties uniformly at random.
"""
function argmin_ties(itr,rng=Random.default_rng())
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

"""
    labeling2predprog(n, FX::Union{CovariatesCopula, CovariatesIndependent}, labeling)

Return the predictive and prognostic covariates from a `labeling` given the covariates generation given by FX.
"""
function labeling2predprog(n, FX::Union{CovariatesCopula, CovariatesIndependent}, labeling)
    partition = covariates_partition(FX)
    labeling2predprog(n, length(FX), labeling, partition)
end

"""
    labeling2predprog(n, m, labeling, partition=[[i] for i in 2:m])

Return the predictive and prognostic covariates from a `labeling` given the number
of covariates `m` and the partition of covariates `partition`.
"""
function labeling2predprog(n, m, labeling, partition=[[i] for i in 2:m])
    length(labeling) == (n+1)*m || throw(ArgumentError("length of labeling must be equal to (n+1)*m"))
    intercept = !(1 in vcat(partition...))
    sort(vcat(partition...)) == (1+intercept):m || throw(ArgumentError("partition must be a partition of 2:m (if there is an intercept) or 1:m (if there is no intercept)"))
    predictive_bool = falses(m)
    prognostic_bool = falses(m)
    for j in (1+intercept):m
        if labeling[indexFromTreatmentCovariate(0,j,m)]
            prognostic_bool[j] = true
        end
        for w in 1:n
            if labeling[indexFromTreatmentCovariate(w,j,m)]
                predictive_bool[j] = true
                break
            end
        end
    end
    predictive = findall(predictive_bool)
    prognostic = Vector{Vector{Int}}(undef, 0)
    for cov in partition
        if !any(predictive_bool[cov]) # if there are no predictive
            if any(prognostic_bool[cov])
                push!(prognostic, copy(cov))
            end
        else
            prog_candidate = Vector{Int}(undef, 0)
            for j in cov
                if !predictive_bool[j]
                    push!(prog_candidate, j)
                end
            end
            if !isempty(prog_candidate)
                push!(prognostic, prog_candidate)
            end
        end
    end
    return predictive, prognostic
end