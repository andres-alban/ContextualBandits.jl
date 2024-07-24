"""
    SimulationRecorder

Abstract type for simulation recorders used in [simulation_stochastic](@ref).
"""
abstract type SimulationRecorder end


"""
    StandardRecorder

Default recorder used in [simulation_stochastic](@ref).
It records the online and offline regret and the Probability of Incorrect Selection (PICS) for each time step.
`regret_on` and `PICS_on` refer to losses during the trial (online reward losses), 
while `regret_off` and `PICS_off` refer to losses post-trial (offline rewards).

Xregret_on, XPICS_on, Xregret_off, and XPICS_off are the same as regret_on, PICS_on, regret_off, PICS_off but for patients with the covariates of interest.

Some additional metrics are also recorded, such as the fraction of patients treated with each treatment, the cumulative regret and PICS, and the fraction of patients with each covariate of interest treated with each treatment.
"""
mutable struct StandardRecorder <: SimulationRecorder
    regret_on::Vector{Float64}
    cumulregret_on::Vector{Float64}
    PICS_on::Vector{Float64}
    cumulPICS_on::Vector{Float64}
    Wfrac_on::Matrix{Float64}

    regret_off::Vector{Float64}
    PICS_off::Vector{Float64}
    Wfrac_off::Matrix{Float64}

    Xregret_on::Matrix{Float64}
    XPICS_on::Matrix{Float64}
    Xfrac_on::Matrix{Float64}
    XWfrac_on::Array{Float64,3}

    Xregret_off::Matrix{Float64}
    XPICS_off::Matrix{Float64}
    XWfrac_off::Array{Float64,3}

    labeling_frac::Vector{Float64}
    sum_labeling::Vector{Float64}
    # helper variables
    T::Int
    n::Int
    delay::Int
    max_mean_outcome_post::Vector{Float64}
    Xmax_mean_outcome_post::Vector{Float64}
    X_post_weights::Vector{Float64}
    function StandardRecorder()
        new()
    end
end

function initialize!(sr::StandardRecorder, T, n, m, delay, n_post, X_post_weights, n_interest)
    @assert length(X_post_weights) == n_post "length(X_post_weights)=$(length(X_post_weights)) is not consistent with n_post=$n_post"
    sum(X_post_weights) ≈ 1.0 || @warn "sum(X_post_weights)=$(sum(X_post_weights)) is not equal to 1.0"
    sr.T = T
    sr.n = n
    sr.delay = delay
    sr.X_post_weights = X_post_weights
    sr.regret_on = Vector{Float64}(undef,T)
    sr.cumulregret_on = Vector{Float64}(undef,T)
    sr.PICS_on = Vector{Float64}(undef,T)
    sr.cumulPICS_on = Vector{Float64}(undef,T)
    sr.Wfrac_on = Matrix{Float64}(undef,T,n)

    sr.regret_off = Vector{Float64}(undef,T+1)
    sr.PICS_off = Vector{Float64}(undef,T+1)
    sr.Wfrac_off = Matrix{Float64}(undef,T+1,n)

    sr.Xregret_on = Matrix{Float64}(undef,T,n_interest)
    sr.XPICS_on = Matrix{Float64}(undef,T,n_interest)
    sr.Xfrac_on = Matrix{Float64}(undef,T,n_interest)
    sr.XWfrac_on = Array{Float64,3}(undef,T,n,n_interest)

    sr.Xregret_off = Matrix{Float64}(undef,T+1,n_interest)
    sr.XPICS_off = Matrix{Float64}(undef,T+1,n_interest)
    sr.XWfrac_off = Array{Float64,3}(undef,T+1,n,n_interest)

    sr.labeling_frac = Vector{Float64}(undef,(n+1)*m)
    sr.sum_labeling = Vector{Float64}(undef,T+1)

    sr.max_mean_outcome_post = Vector{Float64}(undef,n_post)
    sr.Xmax_mean_outcome_post = Vector{Float64}(undef,n_interest)

    return sr
end

function reset!(sr::StandardRecorder,outcome_model,X_post,Xinterest)
    for k in axes(X_post,2)
        sr.max_mean_outcome_post[k] = maximum([mean_outcome(outcome_model,iw,view(X_post,:,k)) for iw in 1:sr.n])
    end
    for k in 1:size(Xinterest,2)
        sr.Xmax_mean_outcome_post[k] = maximum([mean_outcome(outcome_model,iw,view(Xinterest,:,k)) for iw in 1:sr.n])
    end
    sr.regret_off .= 0.0
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
        mu = Vector{Float64}(undef,sr.n)
        for i in 1:sr.n
            mu[i] = mean_outcome(outcome_model,i,Xcurrent)
        end
        for k in axes(Xinterest,2)
            if Xcurrent == view(Xinterest,:,k)
                sr.Xfrac_on[t,k] = 1.0
            end
        end
        best_treat = maximum(mu)
        sr.Wfrac_on[t,Wcurrent] = 1.0
        sr.regret_on[t] = best_treat - mu[Wcurrent]
        sr.PICS_on[t] = sr.regret_on[t] != 0.0
        for k in axes(Xinterest,2)
            if Xcurrent == view(Xinterest,:,k)
                sr.XWfrac_on[t,Wcurrent,k] = 1.0
                sr.Xregret_on[t,k] = sr.regret_on[t]
                sr.XPICS_on[t,k] = sr.PICS_on[t]
            else
                sr.Xregret_on[t,k] = NaN
                sr.XPICS_on[t,k] = NaN
            end
        end
        if t > 1
            sr.cumulregret_on[t] = sr.cumulregret_on[t-1] + sr.regret_on[t]
            sr.cumulPICS_on[t] = sr.cumulPICS_on[t-1] + sr.PICS_on[t]
        else # t=0
            sr.cumulregret_on[t] = sr.regret_on[t]
            sr.cumulPICS_on[t] = sr.PICS_on[t]
        end
    end

    if t >= sr.delay
        treat_post = implementation(policy,X_post,W,X,Y)
        for k in axes(X_post,2) # loop over patients post-trial
            x = sr.max_mean_outcome_post[k] - mean_outcome(outcome_model,treat_post[k],view(X_post,:,k))
            sr.regret_off[t+1] += sr.X_post_weights[k] * x
            sr.PICS_off[t+1] += sr.X_post_weights[k] * (x > 0)
            sr.Wfrac_off[t+1,treat_post[k]] += sr.X_post_weights[k]
        end

        Xtreat_post = implementation(policy,Xinterest,W,X,Y)
        for k in 1:size(Xinterest,2) # loop over patients with Xinterest covariates
            x = sr.Xmax_mean_outcome_post[k] - mean_outcome(outcome_model,Xtreat_post[k],view(Xinterest,:,k))
            sr.Xregret_off[t+1,k] = x
            sr.XPICS_off[t+1,k] = (x > 0)
            sr.XWfrac_off[t+1,Xtreat_post[k],k] = 1
        end

        # compute the number of active terms in labeling
        sr.sum_labeling[t+1] = try sum(policy_labeling(policy)) catch; 0 end
    end

    if t == sr.T+sr.delay
        sr.labeling_frac .= try policy_labeling(policy) catch; 0 end
    end

    return sr
end

function output_recorder(sr::StandardRecorder)
    return (
        sr.regret_on,
        sr.cumulregret_on,
        sr.PICS_on,
        sr.cumulPICS_on,
        sr.Wfrac_on,

        sr.regret_off,
        sr.PICS_off,
        sr.Wfrac_off,

        sr.Xregret_on,
        sr.XPICS_on,
        sr.Xfrac_on,
        sr.XWfrac_on,

        sr.Xregret_off,
        sr.XPICS_off,
        sr.XWfrac_off,

        sr.labeling_frac,
        sr.sum_labeling
    )
end

function output_recorder_names(sr::StandardRecorder)
    return [
        "regret_on",
        "cumulregret_on",
        "PICS_on",
        "cumulPICS_on",
        "Wfrac_on",

        "regret_off",
        "PICS_off",
        "Wfrac_off",

        "Xregret_on",
        "XPICS_on",
        "Xfrac_on",
        "XWfrac_on",

        "Xregret_off",
        "XPICS_off",
        "XWfrac_off",

        "labeling_frac",
        "sum_labeling"
    ]
end