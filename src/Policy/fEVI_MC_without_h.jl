function fEVI_MC_without_h(n, m, theta, Sigma, sample_std, Xt, FXtilde, Wpipeline, Xpipeline, etaon, etaoff, labeling=vcat(falses(m),trues(n*m)), rng=Random.default_rng())
    theta_temp = similar(theta)
    nu = zeros(n)
    WkronX_pipe = interact(Wpipeline,n,Xpipeline,labeling)
    pipelength = length(Wpipeline)
    X_post = Array{Float64,2}(undef,length(FXtilde),etaoff)
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
        sigmatilde[:,w] = Sigma_temp * WkronX ./ sqrt(sample_std + WkronX' * Sigma_temp * WkronX)
    end

    for _ in 1:etaon
        if pipelength > 0
            randn!(rng,Z)
            theta_temp .= theta + sigmatilde_pipe * Z
        else
            theta_temp .= theta
        end
        Z_post = randn(rng)
        rand!(rng,FXtilde,X_post)
        for w in 1:n
            theta_post = theta_temp + sigmatilde[:,w] * Z_post
            for k in 1:etaoff
                # if you want to compare the output of this fuction with the output of the fEVI function, you should uncomment the subtraction to normalize
                nu[w] += maximum([interact(iw,n,view(X_post,:,k), labeling)' * theta_post for iw in 1:n]) - maximum([interact(iw,n,view(X_post,:,k), labeling)' * theta_temp for iw in 1:n])
            end
        end
    end
    nu ./= etaon*etaoff
    return nu
end

function fEVI_MC_without_h_indep(n, m, theta, Sigma, sample_std, Xt, FXtilde, Wpipeline, Xpipeline, etaon, etaoff, labeling=vcat(falses(m),trues(n*m)), rng=Random.default_rng())
    theta_temp = similar(theta)
    nu = zeros(n)
    WkronX_pipe = interact(Wpipeline,n,Xpipeline,labeling)
    pipelength = length(Wpipeline)
    X_post = Array{Float64,2}(undef,length(FXtilde),etaoff)
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
        sigmatilde[:,w] = Sigma_temp * WkronX ./ sqrt(sample_std + WkronX' * Sigma_temp * WkronX)
    end

    for _ in 1:etaon
        for w in 1:n
            if pipelength > 0
                randn!(rng,Z)
                theta_temp .= theta + sigmatilde_pipe * Z
            else
                theta_temp .= theta
            end
            Z_post = randn(rng)
            theta_post = theta_temp + sigmatilde[:,w] * Z_post
            rand!(rng,FXtilde,X_post)
            for k in 1:etaoff
                # if you want to compare the output of this fuction with the output of the fEVI function, you should uncomment the subtraction to normalize
                nu[w] += maximum([interact(iw,n,view(X_post,:,k), labeling)' * theta_post for iw in 1:n]) - maximum([interact(iw,n,view(X_post,:,k), labeling)' * theta_temp for iw in 1:n])
            end
        end
    end
    nu ./= etaon*etaoff
    return nu
end
