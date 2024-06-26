"""
    SimulationRecorder

Abstract type for simulation recorders used in [simulation_stochastic](@ref).
"""
abstract type SimulationRecorder end


"""
    StandardRecorder

Default recorder used in [simulation_stochastic](@ref).
It records the Expected Opportunity Cost (EOC) and the Probability of Incorrect Selection (PICS) for each time step.
`EOC_on` and `PICS_on` refer to losses during the trial (online reward losses), 
while `EOC_off` and `PICS_off` refer to losses post-trial (offline rewards).

XEOC_on, XPICS_on, XEOC_off, and XPICS_off are the same as EOC_on, PICS_on, EOC_off, PICS_off but for patients with the covariates of interest.

Some additional metrics are also recorded, such as the fraction of patients treated with each treatment, the cumulative EOC and PICS, and the fraction of patients with each covariate of interest treated with each treatment.
"""
mutable struct StandardRecorder <: SimulationRecorder
    EOC_on::Vector{Float64}
    cumulEOC_on::Vector{Float64}
    PICS_on::Vector{Float64}
    cumulPICS_on::Vector{Float64}
    Wfrac_on::Matrix{Float64}

    EOC_off::Vector{Float64}
    PICS_off::Vector{Float64}
    Wfrac_off::Matrix{Float64}

    XEOC_on::Matrix{Float64}
    XPICS_on::Matrix{Float64}
    Xfrac_on::Matrix{Float64}
    XWfrac_on::Array{Float64,3}

    XEOC_off::Matrix{Float64}
    XPICS_off::Matrix{Float64}
    XWfrac_off::Array{Float64,3}

    labelsfrac::Vector{Float64}
    Nactivelabels::Vector{Float64}
    # helper variables
    T::Int
    Wn::Int
    delay::Int
    max_mean_outcome_post::Vector{Float64}
    Xmax_mean_outcome_post::Vector{Float64}
    X_post_weights::Vector{Float64}
    function StandardRecorder()
        new()
    end
end

function initialize!(sr::StandardRecorder, T, Wn, m, delay, n_post, X_post_weights, n_interest)
    @assert length(X_post_weights) == n_post "length(X_post_weights)=$(length(X_post_weights)) is not consistent with n_post=$n_post"
    sum(X_post_weights) ≈ 1.0 || @warn "sum(X_post_weights)=$(sum(X_post_weights)) is not equal to 1.0"
    sr.T = T
    sr.Wn = Wn
    sr.delay = delay
    sr.X_post_weights = X_post_weights
    sr.EOC_on = Vector{Float64}(undef,T)
    sr.cumulEOC_on = Vector{Float64}(undef,T)
    sr.PICS_on = Vector{Float64}(undef,T)
    sr.cumulPICS_on = Vector{Float64}(undef,T)
    sr.Wfrac_on = Matrix{Float64}(undef,T,Wn)

    sr.EOC_off = Vector{Float64}(undef,T+1)
    sr.PICS_off = Vector{Float64}(undef,T+1)
    sr.Wfrac_off = Matrix{Float64}(undef,T+1,Wn)

    sr.XEOC_on = Matrix{Float64}(undef,T,n_interest)
    sr.XPICS_on = Matrix{Float64}(undef,T,n_interest)
    sr.Xfrac_on = Matrix{Float64}(undef,T,n_interest)
    sr.XWfrac_on = Array{Float64,3}(undef,T,Wn,n_interest)

    sr.XEOC_off = Matrix{Float64}(undef,T+1,n_interest)
    sr.XPICS_off = Matrix{Float64}(undef,T+1,n_interest)
    sr.XWfrac_off = Array{Float64,3}(undef,T+1,Wn,n_interest)

    sr.labelsfrac = Vector{Float64}(undef,(Wn+1)*m)
    sr.Nactivelabels = Vector{Float64}(undef,T+1)

    sr.max_mean_outcome_post = Vector{Float64}(undef,n_post)
    sr.Xmax_mean_outcome_post = Vector{Float64}(undef,n_interest)

    return sr
end

function reset!(sr::StandardRecorder,outcome_model,X_post,Xinterest)
    for k in axes(X_post,2)
        sr.max_mean_outcome_post[k] = maximum([mean_outcome(outcome_model,iw,view(X_post,:,k)) for iw in 1:sr.Wn])
    end
    for k in 1:size(Xinterest,2)
        sr.Xmax_mean_outcome_post[k] = maximum([mean_outcome(outcome_model,iw,view(Xinterest,:,k)) for iw in 1:sr.Wn])
    end
    sr.EOC_off .= 0.0
    sr.PICS_off .= 0.0

    sr.Wfrac_on .= 0.0
    sr.Wfrac_off .= 0.0

    sr.Xfrac_on .= 0.0
    sr.XWfrac_on .= 0.0

    sr.XWfrac_off .= 0.0

    return sr
end

function record!(sr::StandardRecorder,t,outcome_model,policy,Wcurrent,Xcurrent,Xinterest,X_post,W,X,Y)
    if 1 <= t <= sr.T
        mu = Vector{Float64}(undef,sr.Wn)
        for i in 1:sr.Wn
            mu[i] = mean_outcome(outcome_model,i,Xcurrent)
        end
        for k in axes(Xinterest,2)
            if Xcurrent == view(Xinterest,:,k)
                sr.Xfrac_on[t,k] = 1.0
            end
        end
        best_treat = maximum(mu)
        sr.Wfrac_on[t,Wcurrent] = 1.0
        sr.EOC_on[t] = best_treat - mu[Wcurrent]
        sr.PICS_on[t] = sr.EOC_on[t] != 0.0
        for k in axes(Xinterest,2)
            if Xcurrent == view(Xinterest,:,k)
                sr.XWfrac_on[t,Wcurrent,k] = 1.0
                sr.XEOC_on[t,k] = sr.EOC_on[t]
                sr.XPICS_on[t,k] = sr.PICS_on[t]
            else
                sr.XEOC_on[t,k] = NaN
                sr.XPICS_on[t,k] = NaN
            end
        end
        if t > 1
            sr.cumulEOC_on[t] = sr.cumulEOC_on[t-1] + sr.EOC_on[t]
            sr.cumulPICS_on[t] = sr.cumulPICS_on[t-1] + sr.PICS_on[t]
        else # t=0
            sr.cumulEOC_on[t] = sr.EOC_on[t]
            sr.cumulPICS_on[t] = sr.PICS_on[t]
        end
    end

    if t >= sr.delay
        treat_post = implementation(policy,X_post,W,X,Y)
        for k in axes(X_post,2) # loop over patients post-trial
            x = sr.max_mean_outcome_post[k] - mean_outcome(outcome_model,treat_post[k],view(X_post,:,k))
            sr.EOC_off[t+1] += sr.X_post_weights[k] * x
            sr.PICS_off[t+1] += sr.X_post_weights[k] * (x > 0)
            sr.Wfrac_off[t+1,treat_post[k]] += sr.X_post_weights[k]
        end

        Xtreat_post = implementation(policy,Xinterest,W,X,Y)
        for k in 1:size(Xinterest,2) # loop over patients with Xinterest covariates
            x = sr.Xmax_mean_outcome_post[k] - mean_outcome(outcome_model,Xtreat_post[k],view(Xinterest,:,k))
            sr.XEOC_off[t+1,k] = x
            sr.XPICS_off[t+1,k] = (x > 0)
            sr.XWfrac_off[t+1,Xtreat_post[k],k] = 1
        end

        # compute the number of active labels
        sr.Nactivelabels[t+1] = try sum(policy_labeling(policy)) catch; 0 end
    end

    if t == sr.T+sr.delay
        sr.labelsfrac .= try policy_labeling(policy) catch; 0 end
    end

    return sr
end

function output_recorder(sr::StandardRecorder)
    return (
        sr.EOC_on,
        sr.cumulEOC_on,
        sr.PICS_on,
        sr.cumulPICS_on,
        sr.Wfrac_on,

        sr.EOC_off,
        sr.PICS_off,
        sr.Wfrac_off,

        sr.XEOC_on,
        sr.XPICS_on,
        sr.Xfrac_on,
        sr.XWfrac_on,

        sr.XEOC_off,
        sr.XPICS_off,
        sr.XWfrac_off,

        sr.labelsfrac,
        sr.Nactivelabels
    )
end

function output_recorder_names(sr::StandardRecorder)
    return [
        "EOC_on",
        "cumulEOC_on",
        "PICS_on",
        "cumulPICS_on",
        "Wfrac_on",

        "EOC_off",
        "PICS_off",
        "Wfrac_off",

        "XEOC_on",
        "XPICS_on",
        "Xfrac_on",
        "XWfrac_on",

        "XEOC_off",
        "XPICS_off",
        "XWfrac_off",

        "labelsfrac",
        "Nactivelabels"
    ]
end