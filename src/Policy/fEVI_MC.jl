function fEVI_MC(n, m, theta, Sigma, sample_std, Xt, FXtilde, Wpipeline, Xpipeline, etaon, etaoff, labeling=vcat(falses(m),trues(n*m)), rng=Random.default_rng())
    # length(FXtilde) == m || throw
    # sum(labeling) == length(theta) || throw
    theta_temp = similar(theta)
    lognu = ones(n) * -Inf
    WkronX_pipe = interact(Wpipeline,n,Xpipeline,labeling)
    pipelength = length(Wpipeline)
    X_post = Matrix{Float64}(undef,length(FXtilde),etaoff)
    Z = Vector{Float64}(undef,size(WkronX_pipe,2))
    if pipelength > 0
        sigmatilde_pipe = Sigma * WkronX_pipe * cholesky(inv(Symmetric(sample_std*I + WkronX_pipe' * Sigma * WkronX_pipe))).L
        _,Sigma_temp = BayesUpdateNormal(theta,Sigma,WkronX_pipe,zeros(pipelength),sample_std)
    else
        Sigma_temp = Sigma
    end
    WkronX = Vector{Float64}(undef,length(theta))
    sigmatilde = Matrix{Float64}(undef,length(theta),n)
    for w in 1:n
        interact!(WkronX,w,n,Xt,labeling)
        s = WkronX' * Sigma_temp * WkronX
        if s < 0
            println(Sigma_temp)
            sigmatilde[:,w] .= 0
        else
            sigmatilde[:,w] = Sigma_temp * WkronX ./ sqrt(sample_std + s)
        end
    end

    for _ in 1:etaon
        if pipelength > 0
            randn!(rng,Z)
            theta_temp .= theta + sigmatilde_pipe * Z
        else
            theta_temp .= theta
        end
        rand!(rng,FXtilde,X_post)
        for k in 1:etaoff
            for w in 1:n
                a = [interact(i,n,view(X_post,:,k),labeling)' * theta_temp for i in 1:n]
                b = [interact(i,n,view(X_post,:,k),labeling)' * sigmatilde[:,w] for i in 1:n]
                logQ = logEmaxAffine(a,b)
                lognu[w] = logSumExp(lognu[w],logQ)
            end
        end
    end
    lognu .-= log(etaon*etaoff)
    return lognu
end

function fEVI_MC_indep(n, m, theta, Sigma, sample_std, Xt, FXtilde, Wpipeline, Xpipeline, etaon, etaoff, labeling=vcat(falses(m),trues(n*m)), rng=Random.default_rng())
    theta_temp = similar(theta)
    lognu = ones(n) * -Inf
    WkronX_pipe = interact(Wpipeline,n,Xpipeline,labeling)
    pipelength = length(Wpipeline)
    X_post = Matrix{Float64}(undef,length(FXtilde),etaoff)
    Z = Vector{Float64}(undef,size(WkronX_pipe,2))
    if pipelength > 0
        sigmatilde_pipe = Sigma * WkronX_pipe * cholesky(inv(Symmetric(sample_std*I + WkronX_pipe' * Sigma * WkronX_pipe))).L
        _,Sigma_temp = BayesUpdateNormal(theta,Sigma,WkronX_pipe,zeros(pipelength),sample_std)
    else
        Sigma_temp = Sigma
    end
    WkronX = Vector{Float64}(undef,length(theta))
    sigmatilde = Matrix{Float64}(undef,length(theta),n)
    for w in 1:n
        interact!(WkronX,w,n,Xt,labeling)
        s = WkronX' * Sigma_temp * WkronX
        if s < 0
            sigmatilde[:,w] .= 0
        else
            sigmatilde[:,w] = Sigma_temp * WkronX ./ sqrt(sample_std + s)
        end
    end

    for _ in 1:etaon
        for w in 1:n
            if pipelength > 0
                randn!(rng,Z)
                theta_temp .= theta + sigmatilde_pipe * Z
            else
                theta_temp .= theta
            end
            rand!(rng,FXtilde,X_post)
            for k in 1:etaoff
                a = [interact(i,n,view(X_post,:,k),labeling)' * theta_temp for i in 1:n]
                b = [interact(i,n,view(X_post,:,k),labeling)' * sigmatilde[:,w] for i in 1:n]
                logQ = logEmaxAffine(a,b)
                lognu[w] = logSumExp(lognu[w],logQ)
            end
        end
    end
    lognu .-= log(etaon*etaoff)
    return lognu
end


"""
    fEVI_MC_PolicyLinear <: PolicyLinear
    fEVI_MC_PolicyLinear(n, m, theta0, Sigma0, sample_std, FXtilde, etaon, etaoff, labeling=vcat(falses(m),trues(n*m)))

Allocate treatment using the fEVI-MC allocation policy and update based on the
linear model with labeling to make an implementation.
"""
mutable struct fEVI_MC_PolicyLinear <: PolicyLinear
    model::BayesLinearRegression
    FXtilde::Sampleable
    etaon::Int
    etaoff::Int
end

function fEVI_MC_PolicyLinear(n, m, theta0, Sigma0, sample_std, FXtilde, etaon, etaoff, labeling=vcat(falses(m),trues(n*m)))
    length(FXtilde) == m || throw(DomainError(FXtilde,"`FXtilde` must have length `m`."))
    etaon > 0 || throw(DomainError(etaon,"`etaon` must be positive."))
    etaoff > 0 || throw(DomainError(etaoff,"`etaoff` must be positive."))
    fEVI_MC_PolicyLinear(BayesLinearRegression(n, m, theta0, Sigma0, sample_std, labeling), FXtilde, etaon, etaoff)
end

function allocation(policy::fEVI_MC_PolicyLinear, Xcurrent, W, X, Y, rng=Random.default_rng())
    pipeend = length(W)
    pipestart = length(Y) + 1
    fEVI_MC_indices = fEVI_MC(policy.model.n, policy.model.m, policy.model.theta_t, policy.model.Sigma_t, policy.model.sample_std, Xcurrent,
        policy.FXtilde, view(W,pipestart:pipeend), view(X,:,pipestart:pipeend), policy.etaon, policy.etaoff, policy.model.labeling, rng)
    return argmax_ties(fEVI_MC_indices, rng)
end