"""
    default_prior_linear(n,m,sigma0,psi,D,labeling=vcat(falses(m),trues(n*m)))

Return the default prior (prior mean and covariance matrix) for a linear model 
with `n` treatments and `m` covariates. `sigma0` is the prior standard deviation.
`psi` is the decay parameter for the covariance between coefficients of the same covariate.
`D` is the symmetric distance matrix between treatments, which is of size `(n,n)`.

# Example
```julia
n = 3
m = 3
sigma0 = 1.0
psi = log(2)
D = [0 1 2;
    1 0 1;
    2 1 0]
labeling = BitVector([0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
theta0, Sigma0 = default_prior_linear(n, m, sigma0, psi, D, labeling)
```
"""
function default_prior_linear(n,m,sigma0,psi,D,labeling=vcat(falses(m),trues(n*m)))
    size(D) == (n,n) || throw(ArgumentError("The distance matrix D must be of dimension (n,n) = ($(n),$(n)) instead of $(size(D))"))
    issymmetric(D) || throw(ArgumentError("The distance matrix D must be symmetric"))
    length(labeling) == (n+1)*m || throw(DomainError(labeling,"`labeling` must have length `(n+1)*m`."))
    d = sum(labeling)
    theta0 = zeros(d)
    Sigma0 = zeros(d,d)
    indices = findall(labeling)
    for i in eachindex(indices)
        index = indices[i]
        for j in eachindex(indices)
            index2 = indices[j]
            if i == j
                Sigma0[i,j] = sigma0^2
            elseif treatmentFromIndex(index,m) == 0 || treatmentFromIndex(index2,m) == 0
                Sigma0[i,j] = 0
            elseif covariateFromIndex(index,m) == covariateFromIndex(index2,m)
                Sigma0[i,j] = sigma0^2*exp(-psi*D[treatmentFromIndex(index,m),treatmentFromIndex(index2,m)])
            end
        end
    end
    return theta0, Sigma0
end

"""
    robustify_prior_linear!(theta, Sigma, n, m, labeling=vcat(falses(m),trues(n*m)), z_alpha=2, c=4)

Robustify the prior mean `theta` and covariance matrix `Sigma` for a linear model with `n` treatments and `m` covariates.
"""
function robustify_prior_linear!(theta, Sigma, n, m, labeling=vcat(falses(m),trues(n*m)), z_alpha=2, c=4)
    length(labeling) == (n+1)*m || throw(DomainError(labeling,"`labeling` must have length `(n+1)*m`."))
    d = sum(labeling)
    length(theta) == d || throw(DomainError(theta0,"`theta0` must be of length `sum(labeling)`."))
    size(Sigma) == (d,d) || throw(DomainError(Sigma0,"`Sigma0` must be of dimensions `sum(labeling)`."))

    indices = findall(labeling)
    # find the maximum theta and sigma stratified by covariate
    maxtheta_by_covariate = -Inf*ones(m)
    maxsigma_by_covariate = -Inf*ones(m)
    for i in eachindex(indices)
        cov = covariateFromIndex(indices[i],m)
        if theta[i] > maxtheta_by_covariate[cov]
            maxtheta_by_covariate[cov] = theta[i]
        end
        if Sigma[i,i] > maxsigma_by_covariate[cov]
            maxsigma_by_covariate[cov] = Sigma[i,i]
        end
    end
    # Robustify the prior
    for i in eachindex(indices)
        if treatmentFromIndex(indices[i],m) == 0
            theta[i] = 0
        else
            cov = covariateFromIndex(indices[i],m)
            theta[i] = maxtheta_by_covariate[cov] + z_alpha * maxsigma_by_covariate[cov]
        end
    end
    Sigma .= c .* Sigma
    return theta, Sigma
end