function fEVI_MC_without_h(n, m, theta, Sigma, sample_std, Xt, FXtilde, Wpipeline, Xpipeline, etaon, etaoff, labeling=vcat(falses(m), trues(n * m)), rng=Random.default_rng())
    theta_temp = similar(theta)
    nu = zeros(n)
    WkronX_pipe = interact(Wpipeline, n, Xpipeline, labeling)
    pipelength = length(Wpipeline)
    X_post = Matrix{Float64}(undef, length(FXtilde), etaoff)
    Z = Vector{Float64}(undef, size(WkronX_pipe, 2))
    if pipelength > 0
        sigmatilde_pipe = Sigma * WkronX_pipe * cholesky(inv(Symmetric(sample_std * I + WkronX_pipe' * Sigma * WkronX_pipe))).L
        _, Sigma_temp = BayesUpdateNormal(theta, Sigma, WkronX_pipe, zeros(pipelength), sample_std)
    else
        Sigma_temp = Sigma
    end
    WkronX = Vector{Float64}(undef, length(theta))
    sigmatilde = Matrix{Float64}(undef, length(theta), n)
    for w in 1:n
        interact!(WkronX, w, n, Xt, labeling)
        sigmatilde[:, w] = Sigma_temp * WkronX ./ sqrt(sample_std + WkronX' * Sigma_temp * WkronX)
    end

    for _ in 1:etaon
        if pipelength > 0
            randn!(rng, Z)
            theta_temp .= theta + sigmatilde_pipe * Z
        else
            theta_temp .= theta
        end
        Z_post = randn(rng)
        rand!(rng, FXtilde, X_post)
        for w in 1:n
            theta_post = theta_temp + sigmatilde[:, w] * Z_post
            for k in 1:etaoff
                # if you want to compare the output of this fuction with the output of the fEVI function, you should uncomment the subtraction to normalize
                nu[w] += maximum([interact(iw, n, view(X_post, :, k), labeling)' * theta_post for iw in 1:n]) - maximum([interact(iw, n, view(X_post, :, k), labeling)' * theta_temp for iw in 1:n])
            end
        end
    end
    nu ./= etaon * etaoff
    return nu
end

function fEVI_MC_without_h_indep(n, m, theta, Sigma, sample_std, Xt, FXtilde, Wpipeline, Xpipeline, etaon, etaoff, labeling=vcat(falses(m), trues(n * m)), rng=Random.default_rng())
    theta_temp = similar(theta)
    nu = zeros(n)
    WkronX_pipe = interact(Wpipeline, n, Xpipeline, labeling)
    pipelength = length(Wpipeline)
    X_post = Matrix{Float64}(undef, length(FXtilde), etaoff)
    Z = Vector{Float64}(undef, size(WkronX_pipe, 2))
    if pipelength > 0
        sigmatilde_pipe = Sigma * WkronX_pipe * cholesky(inv(Symmetric(sample_std * I + WkronX_pipe' * Sigma * WkronX_pipe))).L
        _, Sigma_temp = BayesUpdateNormal(theta, Sigma, WkronX_pipe, zeros(pipelength), sample_std)
    else
        Sigma_temp = Sigma
    end
    WkronX = Vector{Float64}(undef, length(theta))
    sigmatilde = Matrix{Float64}(undef, length(theta), n)
    for w in 1:n
        interact!(WkronX, w, n, Xt, labeling)
        sigmatilde[:, w] = Sigma_temp * WkronX ./ sqrt(sample_std + WkronX' * Sigma_temp * WkronX)
    end

    for _ in 1:etaon
        for w in 1:n
            if pipelength > 0
                randn!(rng, Z)
                theta_temp .= theta + sigmatilde_pipe * Z
            else
                theta_temp .= theta
            end
            Z_post = randn(rng)
            theta_post = theta_temp + sigmatilde[:, w] * Z_post
            rand!(rng, FXtilde, X_post)
            for k in 1:etaoff
                # if you want to compare the output of this fuction with the output of the fEVI function, you should uncomment the subtraction to normalize
                nu[w] += maximum([interact(iw, n, view(X_post, :, k), labeling)' * theta_post for iw in 1:n]) - maximum([interact(iw, n, view(X_post, :, k), labeling)' * theta_temp for iw in 1:n])
            end
        end
    end
    nu ./= etaon * etaoff
    return nu
end


"""
    fEVI_MC_simple_PolicyLinear <: PolicyLinear
    fEVI_MC_simple_PolicyLinear(n, m, theta0, Sigma0, sample_std, FXtilde, etaon, etaoff, labeling=vcat(falses(m),trues(n*m)))

Allocate treatment using the fEVI-MC-simple allocation policy and update based on the
linear model with labeling to make an implementation.
"""
mutable struct fEVI_MC_simple_PolicyLinear <: PolicyLinear
    model::BayesLinearRegression
    FXtilde::Sampleable
    etaon::Int
    etaoff::Int
end

function fEVI_MC_simple_PolicyLinear(n, m, theta0, Sigma0, sample_std, FXtilde, etaon, etaoff, labeling=vcat(falses(m), trues(n * m)))
    length(FXtilde) == m || throw(DomainError(FXtilde, "`FXtilde` must have length `m`."))
    etaon > 0 || throw(DomainError(etaon, "`etaon` must be positive."))
    etaoff > 0 || throw(DomainError(etaoff, "`etaoff` must be positive."))
    fEVI_MC_simple_PolicyLinear(BayesLinearRegression(n, m, theta0, Sigma0, sample_std, labeling), FXtilde, etaon, etaoff)
end

function allocation(policy::fEVI_MC_simple_PolicyLinear, Xcurrent, W, X, Y, rng=Random.default_rng())
    pipeend = length(W)
    pipestart = length(Y) + 1
    fEVI_MC_indices = fEVI_MC_without_h(policy.model.n, policy.model.m, policy.model.theta_t, policy.model.Sigma_t, policy.model.sample_std, Xcurrent,
        policy.FXtilde, view(W, pipestart:pipeend), view(X, :, pipestart:pipeend), policy.etaon, policy.etaoff, policy.model.labeling, rng)
    return argmax_ties(fEVI_MC_indices, rng)
end



"""
    fEVI_MC_OnOff_simple_PolicyLinear <: PolicyLinear
    fEVI_MC_OnOff_simple_PolicyLinear(n, m, theta0, Sigma0, sample_std, FXtilde, etaon, etaoff, T, delay, P[, labeling])

Allocate treatment using the fEVI-MC-simple with online rewards allocation policy and
update based on the linear model with labeling to make an implementation.
"""
mutable struct fEVI_MC_OnOff_simple_PolicyLinear <: PolicyLinear
    model::BayesLinearRegression
    FXtilde::Sampleable
    etaon::Int
    etaoff::Int
    T::Int
    delay::Int
    P::Float64
end

function fEVI_MC_OnOff_simple_PolicyLinear(n, m, theta0, Sigma0, sample_std, FXtilde, etaon, etaoff, T, delay, P, labeling=vcat(falses(m), trues(n * m)))
    length(FXtilde) == m || throw(DomainError(FXtilde, "`FXtilde` must have length `m`."))
    etaon > 0 || throw(DomainError(etaon, "`etaon` must be positive."))
    etaoff > 0 || throw(DomainError(etaoff, "`etaoff` must be positive."))
    T >= 0 || throw(DomainError(T, "`T` must be positive"))
    delay >= 0 || throw(DomainError(delay, "`delay` must be positive"))
    P >= 0 || throw(DomainError(P, "`P` must be positive"))
    fEVI_MC_OnOff_simple_PolicyLinear(BayesLinearRegression(n, m, theta0, Sigma0, sample_std, labeling), FXtilde, etaon, etaoff, T, delay, P)
end

function allocation(policy::fEVI_MC_OnOff_simple_PolicyLinear, Xcurrent, W, X, Y, rng=Random.default_rng())
    t = length(W)
    pipestart = length(Y) + 1
    weight_on = max(policy.T - t - policy.delay - 1, 0)
    expected_outcomes = [interact(iw, policy.model.n, Xcurrent, policy.model.labeling)' * policy.model.theta_t for iw in 1:policy.model.n]
    expected_outcomes .-= minimum(expected_outcomes) - 1 # shift expected outcomes so that the worst is 1
    log_expected_outcomes = log.(expected_outcomes)
    lognu = fEVI_MC_without_h(policy.model.n, policy.model.m, policy.model.theta_t, policy.model.Sigma_t, policy.model.sample_std, Xcurrent,
        policy.FXtilde, view(W, pipestart:t), view(X, :, pipestart:t), policy.etaon, policy.etaoff, policy.model.labeling, rng)
    lognu_on = logSumExp.(log_expected_outcomes, log(weight_on) .+ lognu)
    lognu_combined = logSumExp.(lognu_on, log(policy.P) .+ lognu)
    return argmax_ties(lognu_combined, rng)
end