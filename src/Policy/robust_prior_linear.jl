"""
    default_prior_linear(n, m, sigma0, psi[, D, labeling])

The default prior is a method to specify a prior for a linear model that only depends 
on a few parameters. It returns the prior mean and covariance matrix for a linear model 
with `n` treatments and `m` covariates. 

The prior mean is a vector of zeros.

The prior covariance matrix (`Sigma0`) has `sigma0^2` in all entries of the diagonal.
`psi` is the decay parameter for the covariance
between coefficients of the same covariate and `D` is a symmetric distance matrix between
treatments, which is of size `(n,n)`. Off-diagonal elements of the covariance matrix
that represent the covariance between two predictive coefficient for the same covariate
with respect to two different treatments, say `w1` and `w2`, are given by
`sigma0^2*exp(-psi*D[w1,w2])`.

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
function default_prior_linear(n, m, sigma0, psi, D=Inf * ones(n, n), labeling=vcat(falses(m), trues(n * m)))
    size(D) == (n, n) || throw(ArgumentError("The distance matrix D must be of dimension (n,n) = ($(n),$(n)) instead of $(size(D))"))
    issymmetric(D) || throw(ArgumentError("The distance matrix D must be symmetric"))
    length(labeling) == (n + 1) * m || throw(DomainError(labeling, "`labeling` must have length `(n+1)*m`."))
    d = sum(labeling)
    theta0 = zeros(d)
    Sigma0 = zeros(d, d)
    indices = findall(labeling)
    for i in eachindex(indices)
        index = indices[i]
        for j in eachindex(indices)
            index2 = indices[j]
            if i == j
                Sigma0[i, j] = sigma0^2
            elseif treatmentFromIndex(index, m) == 0 || treatmentFromIndex(index2, m) == 0
                Sigma0[i, j] = 0
            elseif covariateFromIndex(index, m) == covariateFromIndex(index2, m)
                Sigma0[i, j] = sigma0^2 * exp(-psi * D[treatmentFromIndex(index, m), treatmentFromIndex(index2, m)])
            end
        end
    end
    return theta0, Sigma0
end

"""
    robustify_prior_linear!(theta0, Sigma0, n, m[, labeling, z_alpha, c])

Robustify the prior mean vector `theta0` and covariance matrix `Sigma0` for a linear
model with `n` treatments and `m` covariates. `theta0` elements that are predictive
are artificially increased by a factor of `z_alpha` times the square root of the
largest element of `Sigma0`. `Sigma0` is multiplied by scalar `c`.

By default `z_alpha=2` and `c=4`.
"""
function robustify_prior_linear!(theta0, Sigma0, n, m, labeling=vcat(falses(m), trues(n * m)), z_alpha=2, c=4)
    length(labeling) == (n + 1) * m || throw(DomainError(labeling, "`labeling` must have length `(n+1)*m`."))
    d = sum(labeling)
    length(theta0) == d || throw(DomainError(theta0, "`theta0` must be of length `sum(labeling)`."))
    size(Sigma0) == (d, d) || throw(DomainError(Sigma0, "`Sigma0` must be of dimensions `sum(labeling)`."))

    indices = findall(labeling)
    # find the maximum theta0 and sigma stratified by covariate
    maxtheta_by_covariate = -Inf * ones(m)
    maxsigma_by_covariate = -Inf * ones(m)
    for i in eachindex(indices)
        cov = covariateFromIndex(indices[i], m)
        if theta0[i] > maxtheta_by_covariate[cov]
            maxtheta_by_covariate[cov] = theta0[i]
        end
        if Sigma0[i, i] > maxsigma_by_covariate[cov]
            maxsigma_by_covariate[cov] = Sigma0[i, i]
        end
    end
    # Robustify the prior
    for i in eachindex(indices)
        if treatmentFromIndex(indices[i], m) == 0
            theta0[i] = 0
        else
            cov = covariateFromIndex(indices[i], m)
            theta0[i] = maxtheta_by_covariate[cov] + z_alpha * sqrt(maxsigma_by_covariate[cov])
        end
    end
    Sigma0 .= c .* Sigma0
    return theta0, Sigma0
end