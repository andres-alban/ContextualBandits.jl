"""
    replication_stochastic(X,X_post,Z,n,delay,policy,outcome_model,recorder;Xinterest=Matrix{Float64}(undef,size(X,1),0),X_post_weights=ones(size(X_post,2)),rng=Random.default_rng())

Simulate one replication of a trial with covariates `X`, post-trial covariates `X_post`, noise `Z`, number of treatments `n`, delay `delay`, policy `policy`, outcome model `outcome_model`.
This function is not intended to be called directly, but rather through [simulation_stochastic](@ref).
"""
function replication_stochastic(X, X_post, Z, n, delay, policy, outcome_model, recorder; Xinterest=Matrix{Float64}(undef, size(X, 1), 0), X_post_weights=ones(size(X_post, 2)), rng=Random.default_rng())

    @assert size(X, 1) == size(X_post, 1) "X_post is not consistent with the size of X"
    T = size(X, 2)
    @assert length(Z) == T "Z is not consistent with the size of X: should be $T but is $(length(Z))"
    @assert length(X_post_weights) == size(X_post, 2) "X_post_weights is not consistent with the size of X_post"

    # Outcomes of random variables
    Y = zeros(T)
    W = zeros(Int, T)

    # loop over patients in trial (start at zero to account for delay=0)
    for t in 0:(T+delay)
        # Covariates of the newly arrived patient that has not yet been treated
        Xcurrent = 1 <= t <= T ? view(X, :, t) : view(X, :, 1:0)
        # Allocate
        if 1 <= t <= T
            w = allocation(policy, Xcurrent, view(W, 1:(t-1)), view(X, :, 1:(t-1)), view(Y, 1:(t-delay-1)), rng)
            @assert w in 1:n "The policy $(typeof(policy)) made an allocation outside the range 1-$n"
            W[t] = w
            Y[t] = noisy_outcome(outcome_model, w, view(X, :, t), Z[t])
        end

        # Available data at time t
        Wav = view(W, 1:min(t, T))
        Xav = view(X, :, 1:min(t, T))
        Yav = view(Y, 1:(t-delay))

        # Update state of policy
        state_update!(policy, Wav, Xav, Yav, rng)

        # record metrics
        w = 1 <= t <= T ? W[t] : NaN
        record!(recorder, t, outcome_model, policy, w, Xcurrent, Xinterest, X_post, Wav, Xav, Yav)
    end

    return output_recorder(recorder)
end # function
