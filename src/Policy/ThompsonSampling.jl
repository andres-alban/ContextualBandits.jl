"""
    TS_linear(n, theta, Sigma, Xt, labeling=vcat(falses(size(Xt,1)),trues(size(Xt,1)*n)), rng=Random.default_rng())

Thonpson sampling for linear models to allocate treatment to a patient with covariates `Xt`
using the linear model with parameters `theta` and `Sigma` and the `labeling` of the covariates.
"""
function TS_linear(n, theta, Sigma, Xt, labeling=vcat(falses(size(Xt, 1)), trues(size(Xt, 1) * n)), rng=Random.default_rng())
    Sigma_symm = Symmetric(Sigma)
    mu = randnMv(rng, theta, Sigma_symm)
    reward = Vector{Float64}(undef, n)
    for w in 1:n
        reward[w] = interact(w, n, Xt, labeling)' * mu
    end
    return argmax_ties(reward, rng)
end

"""
    TTTS_linear(n, theta, Sigma, Xt, beta=0.5, maxiter=100, labeling=vcat(falses(size(Xt,1)),trues(size(Xt,1)*n)), rng=Random.default_rng())

Top-Two Thompson Sampling for linear models to allocate treatment to a patient with covariates `Xt`
using the linear model with parameters `theta` and `Sigma` and the `labeling` of the covariates.

`beta` is the probability of selecting the same treatment selected by Thompson Sampling.
With probability `1-beta`, continue drawing Thompson samples until a different treatment is recommended.
To prevent excessive computation time, `maxiter` is the maximum number of iterations to find an alternative treatment.

See [Russo D (2020) Simple Bayesian algorithms for best arm identification. Operations Research 68(6)](https://doi.org/10.1287/opre.2019.1911)
"""
function TTTS_linear(n, theta, Sigma, Xt, beta=0.5, maxiter=100, labeling=vcat(falses(size(Xt, 1)), trues(size(Xt, 1) * n)), rng=Random.default_rng())
    Sigma_symm = Symmetric(Sigma)
    mu = randnMv(rng, theta, Sigma_symm)
    reward = Vector{Float64}(undef, n)
    for w in 1:n
        reward[w] = interact(w, n, Xt, labeling)' * mu
    end
    I = argmax_ties(reward, rng)

    if (rand(rng) < beta)
        return I
    else
        for _ in 1:maxiter
            mu = randnMv(rng, theta, Sigma_symm)
            for w in 1:n
                reward[w] = interact(w, n, Xt, labeling)' * mu
            end
            J = argmax_ties(reward, rng)
            if (J == I)
                continue
            else
                return J
            end
        end
        return I # if no alternative is found after maxiter iterations, then just stick to the alternative that is most likely to be the best, i.e. I
    end
end
"""
    TSPolicyLinear <: PolicyLinear
    TSPolicyLinear(n, m, theta0, Sigma0, sample_std[, labeling])

Allocate treatment using Thompson Sampling and update based on the linear model with labeling to make an implementation.
"""
struct TSPolicyLinear <: PolicyLinear
    model::BayesLinearRegression
end

function TSPolicyLinear(n, m, theta0, Sigma0, sample_std, labeling=vcat(falses(m), trues(n * m)))
    TSPolicyLinear(BayesLinearRegression(n, m, theta0, Sigma0, sample_std, labeling))
end

function allocation(policy::TSPolicyLinear, Xcurrent, W, X, Y, rng=Random.default_rng())
    TS_linear(policy.model.n, policy.model.theta_t, policy.model.Sigma_t, Xcurrent, policy.model.labeling, rng)
end

"""
    TTTSPolicyLinear <: PolicyLinear
    TTTSPolicyLinear(n, m, theta0, Sigma0, sample_std, beta, maxiter[, labeling])

Allocate treatment using Top-Two Thompson Sampling and update based on the linear model with labeling to make an implementation.

[Russo D (2020) Simple Bayesian algorithms for best arm identification. Operations Research 68(6)](https://doi.org/10.1287/opre.2019.1911)
"""
struct TTTSPolicyLinear <: PolicyLinear
    model::BayesLinearRegression
    beta::Float64
    maxiter::Int
end

function TTTSPolicyLinear(n, m, theta0, Sigma0, sample_std, beta, maxiter, labeling=vcat(falses(m), trues(n * m)))
    TTTSPolicyLinear(BayesLinearRegression(n, m, theta0, Sigma0, sample_std, labeling), beta, maxiter)
end

function allocation(policy::TTTSPolicyLinear, Xcurrent, W, X, Y, rng=Random.default_rng())
    TTTS_linear(policy.model.n, policy.model.theta_t, policy.model.Sigma_t, Xcurrent, policy.beta, policy.maxiter, policy.model.labeling, rng)
end