"""
    fEVIaux(gt, Wn, gn, theta, Sigma, sample_std)

Compute auxiliary fEVI values that will be averaged over each treatment in [fEVI](@ref).
This function computes the values described in Equation 12 of 
[Alban, A., Chick, S. E., & Zoumpoulis, S. I. (2021). Expected value of information methods for contextual ranking and selection: clinical trials and simulation optimization. In 2021 Winter Simulation Conference (WSC)](https://ieeexplore.ieee.org/abstract/document/9715303)

The model has `gn` groups and `Wn` treatments.
`theta` and `Sigma` are the prior mean and covariance matrix.
`sample_std` is a vector of sampling standard deviations or a scalar is they are all the same.
`gt` is the group of the subject that was enrolled and is to be allocated to a treatment.
The function returns a matrix `logQ` of size `gn x Wn` where `logQ[g,w]` is the
logarithm of the expected value of information gained to treat patients in group g when the trial ends,
if we were to select treatment `w` to treat the patient in group `gt`.
"""
function fEVIaux(Wn, gn, theta, Sigma, sample_std, gt)
    logQ = Matrix{Float64}(undef,gn,Wn)
    for w in 1:Wn
        current = treatment_g2index(w,gt,gn)
        denominator = sqrt(Sigma[current,current] + sample_std[current]^2)
        if denominator == 0
            logQ[:,w] .= -Inf
        else
            sigmatilde = Sigma[:,current]./denominator
            for g in 1:gn
                post = treatment_g2index.(1:Wn,g,gn)
                logQ[g,w] = logEmaxAffine(theta[post],sigmatilde[post])
            end
        end
    end
    return logQ
end

function fEVIaux(Wn, gn, theta, Sigma, sample_std::Number, gt)
    fEVIaux(Wn, gn, theta, Sigma, fill(sample_std,Wn*gn), gt)
end


"""
    fEVI(gt, Wn, gn, p, theta, Sigma, sample_std)

Compute the fEVI values for each treatment.
This function implements Algorithm 1 of 
[Alban, A., Chick, S. E., & Zoumpoulis, S. I. (2021). Expected value of information methods for contextual ranking and selection: clinical trials and simulation optimization. In 2021 Winter Simulation Conference (WSC)](https://ieeexplore.ieee.org/abstract/document/9715303)

The model has `gn` groups and `Wn` treatments.
`theta` and `Sigma` are the prior mean and covariance matrix.
`sample_std` is a vector of sampling standard deviations or a scalar is they are all the same.
`gt` is the group of the subject that was enrolled and is to be allocated to a treatment.
The returned value is a vector `logQ` of size `Wn` where `logQ[w]` is the
fEVI value of treating the patient in group `gt` with treatment `w`.
"""
function fEVI(Wn, gn, theta, Sigma, sample_std, gt, p)
    logQaux = fEVIaux(Wn, gn, theta, Sigma, sample_std, gt)
    logQ = Vector{Float64}(undef,Wn)
    for w in 1:Wn
        logQ[w] = logSumExp(log.(p) + logQaux[:,w])
    end
    return logQ
end


"""
    fEVIDiscrete <: PolicyLinearDiscrete
    fEVIDiscrete(Wn, m, theta0, Sigma0, sample_std, FX, labeling=vcat(falses(m),trues(Wn*m)))

Allocate treatment using the fEVI allocation policy and update based on the linear model with labeling to make an implementation.
"""
mutable struct fEVIDiscrete <: PolicyLinearDiscrete
    Wn::Int
    m::Int
    theta0::Vector{Float64}
    Sigma0::Matrix{Float64}
    sample_std::Float64
    labeling::BitVector
    FX::Union{CovariatesIndependent,CovariatesCopula}
    gn::Int
    p::Vector{Float64}
    theta_t::Vector{Float64}
    Sigma_t::Matrix{Float64}
    function fEVIDiscrete(Wn, m, theta0, Sigma0, sample_std, FX, labeling=vcat(falses(m),trues(Wn*m)))
        checkInputPolicyLinearDiscrete(Wn, m, theta0, Sigma0, sample_std, labeling, FX)
        gn = total_groups(FX)
        new(Wn, m, copy(theta0), copy(Sigma0), sample_std, copy(labeling), FX, gn, X2g_probs(FX), zeros(Wn*gn), zeros(Wn*gn,Wn*gn))
    end
end

function allocation(policy::fEVIDiscrete,Xcurrent,W,X,Y,rng=Random.GLOBAL_RNG)
    g = X2g(Xcurrent, policy.FX)
    return argmax_ties(fEVI(policy.Wn, policy.gn, policy.theta_t, policy.Sigma_t, policy.sample_std, g, policy.p), rng)
end

"""
    fEVIDiscreteOnOff <: PolicyLinearDiscrete
    fEVIDiscreteOnOff(Wn, m, theta0, Sigma0, sample_std, FX, P, T, labeling=vcat(falses(m),trues(Wn*m)))

Allocate treatment using the fEVI allocation policy (with online and offline rewards)
and update based on linear model with labeling to make an implementation.
"""
mutable struct fEVIDiscreteOnOff <: PolicyLinearDiscrete
    Wn::Int
    m::Int
    theta0::Vector{Float64}
    Sigma0::Matrix{Float64}
    sample_std::Float64
    labeling::BitVector
    FX::Union{CovariatesIndependent,CovariatesCopula}
    gn::Int
    p::Vector{Float64}
    P::Float64
    T::Int
    theta_t::Vector{Float64}
    Sigma_t::Matrix{Float64}
    function fEVIDiscreteOnOff(Wn, m, theta0, Sigma0, sample_std, FX, P, T, labeling=vcat(falses(m),trues(Wn*m)))
        checkInputPolicyLinearDiscrete(Wn, m, theta0, Sigma0, sample_std, labeling, FX)
        gn = total_groups(FX)
        new(Wn, m, copy(theta0), copy(Sigma0), sample_std, copy(labeling), FX, gn, X2g_probs(FX), P, T, similar(theta0), similar(Sigma0))
    end
end

function allocation(policy::fEVIDiscreteOnOff,Xcurrent,W,X,Y,rng=Random.GLOBAL_RNG)
    g = X2g(Xcurrent,policy.FX)
    t = length(W)
    expected_outcomes = policy.theta_t[treatment_g2index.(1:policy.Wn, g, policy.gn)]
    expected_outcomes .-= minimum(expected_outcomes) - 1 # shift expected outcomes so that the worst is 1 and we can take the logarithm
    log_expected_outcomes = log.(expected_outcomes)
    lognu = fEVI(policy.Wn, policy.gn, policy.theta_t, policy.Sigma_t, policy.sample_std, g, policy.p)
    lognu_on = logSumExp.(log_expected_outcomes, log(policy.T - t - 1) .+ lognu)
    lognu_combined = logSumExp.(lognu_on, log(policy.P) .+ lognu)
    return argmax_ties(lognu_combined, rng)
end