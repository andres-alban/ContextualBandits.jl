"""
    fEVIaux(gt, n, gn, theta, Sigma, sample_std)

Compute auxiliary fEVI values that will be averaged over each treatment in [fEVI](@ref).
This function computes the values described in Equation 12 of 
[Alban, A., Chick, S. E., & Zoumpoulis, S. I. (2021). Expected value of information methods for contextual ranking and selection: clinical trials and simulation optimization. In 2021 Winter Simulation Conference (WSC)](https://ieeexplore.ieee.org/abstract/document/9715303)

The model has `gn` groups and `n` treatments.
`theta` and `Sigma` are the prior mean and covariance matrix.
`sample_std` is a vector of sampling standard deviations or a scalar is they are all the same.
`gt` is the group of the subject that was enrolled and is to be allocated to a treatment.
The function returns a matrix `logQ` of size `gn x n` where `logQ[g,w]` is the
logarithm of the expected value of information gained to treat patients in group g when the trial ends,
if we were to select treatment `w` to treat the patient in group `gt`.
"""
function fEVIaux(n, gn, theta, Sigma, sample_std, gt)
    logQ = Matrix{Float64}(undef, gn, n)
    for w in 1:n
        current = treatment_g2index(w, gt, gn)
        denominator = sqrt(Sigma[current, current] + sample_std[current]^2)
        if denominator == 0
            logQ[:, w] .= -Inf
        else
            sigmatilde = Sigma[:, current] ./ denominator
            for g in 1:gn
                post = treatment_g2index.(1:n, g, gn)
                logQ[g, w] = logEmaxAffine(theta[post], sigmatilde[post])
            end
        end
    end
    return logQ
end

function fEVIaux(n, gn, theta, Sigma, sample_std::Number, gt)
    fEVIaux(n, gn, theta, Sigma, fill(sample_std, n * gn), gt)
end


"""
    fEVI(n, gn, theta, Sigma, sample_std, gt, p)

Compute the fEVI values for each treatment.
This function implements Algorithm 1 of 
[Alban, A., Chick, S. E., & Zoumpoulis, S. I. (2021). Expected value of information methods for contextual ranking and selection: clinical trials and simulation optimization. In 2021 Winter Simulation Conference (WSC)](https://ieeexplore.ieee.org/abstract/document/9715303)

The model has `gn` groups and `n` treatments.
`theta` and `Sigma` are the prior mean and covariance matrix.
`sample_std` is a vector of sampling standard deviations or a scalar is they are all the same.
`gt` is the group of the subject that was enrolled and is to be allocated to a treatment.
The returned value is a vector `logQ` of size `n` where `logQ[w]` is the
fEVI value of treating the patient in group `gt` with treatment `w`.
"""
function fEVI(n, gn, theta, Sigma, sample_std, gt, p)
    logQaux = fEVIaux(n, gn, theta, Sigma, sample_std, gt)
    logQ = Vector{Float64}(undef, n)
    for w in 1:n
        logQ[w] = logSumExp(log.(p) + logQaux[:, w])
    end
    return logQ
end


"""
    fEVIDiscrete <: PolicyLinearDiscrete
    fEVIDiscrete(n, m, theta0, Sigma0, sample_std, FX, labeling=vcat(falses(m),trues(n*m)))

Allocate treatment using the fEVI allocation policy and update based on the linear model with labeling to make an implementation.
"""
mutable struct fEVIDiscrete <: PolicyLinearDiscrete
    model::BayesLinearRegressionDiscrete
end

function fEVIDiscrete(n, m, theta0, Sigma0, sample_std, FX, labeling=vcat(falses(m), trues(n * m)))
    fEVIDiscrete(BayesLinearRegressionDiscrete(n, m, theta0, Sigma0, sample_std, FX, labeling))
end

function allocation(policy::fEVIDiscrete, Xcurrent, W, X, Y, rng=Random.default_rng())
    g = X2g(Xcurrent, policy.model.FX)
    return argmax_ties(fEVI(policy.model.n, policy.model.gn, policy.model.theta_t, policy.model.Sigma_t, policy.model.sample_std, g, policy.model.p), rng)
end

"""
    fEVIDiscreteOnOff <: PolicyLinearDiscrete
    fEVIDiscreteOnOff(n, m, theta0, Sigma0, sample_std, FX, P, T, labeling=vcat(falses(m),trues(n*m)))

Allocate treatment using the fEVI allocation policy (with online and offline rewards)
and update based on linear model with labeling to make an implementation.
"""
mutable struct fEVIDiscreteOnOff <: PolicyLinearDiscrete
    model::BayesLinearRegressionDiscrete
    P::Float64
    T::Int
end


function fEVIDiscreteOnOff(n, m, theta0, Sigma0, sample_std, FX, P, T, labeling=vcat(falses(m), trues(n * m)))
    fEVIDiscreteOnOff(BayesLinearRegressionDiscrete(n, m, theta0, Sigma0, sample_std, FX, labeling), P, T)
end

function allocation(policy::fEVIDiscreteOnOff, Xcurrent, W, X, Y, rng=Random.default_rng())
    g = X2g(Xcurrent, policy.model.FX)
    t = length(W)
    weight_on = max(policy.T - t - 1, 0)
    expected_outcomes = policy.model.theta_t[treatment_g2index.(1:policy.model.n, g, policy.model.gn)]
    expected_outcomes .-= minimum(expected_outcomes) - 1 # shift expected outcomes so that the worst is 1 and we can take the logarithm
    log_expected_outcomes = log.(expected_outcomes)
    lognu = fEVI(policy.model.n, policy.model.gn, policy.model.theta_t, policy.model.Sigma_t, policy.model.sample_std, g, policy.model.p)
    lognu_on = logSumExp.(log_expected_outcomes, log(weight_on) .+ lognu)
    lognu_combined = logSumExp.(lognu_on, log(policy.P) .+ lognu)
    return argmax_ties(lognu_combined, rng)
end