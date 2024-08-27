"""
    biasedcoin_score(Xcurrent, W, X, n, partition=1:size(Xcurrent,1), weights=ones(length(partition)+2), target_fraction = ones(n) / n)

Calculate the score for the biased coin policy. The score measures the distance
to a balanced allocation of the treatments.
"""
function biasedcoin_score(Xcurrent, W, X, n, partition=[[i] for i in 2:size(Xcurrent, 1)], weights=ones(length(partition) + 2), target_fraction=ones(n) / n)
    counts = [sum(W .== w) for w in 1:n]


    d_overall = distance_squared_from_counts(counts, target_fraction)

    # Marginal distance
    d_marginal = zeros(length(partition), n)
    for i in eachindex(partition)
        # x_{i,j,k} from Pocock and Simon 1975
        # i is the covariate
        # j are the possible levels of the covariate
        # k is the treatment: k=1:n
        Xcovariate = view(X, partition[i], :)
        xijk = zeros(Int, n)
        for patient in axes(Xcovariate, 2)
            if Xcovariate[:, patient] == Xcurrent[partition[i]]
                xijk[W[patient]] += 1
            end
        end

        d_marginal[i, :] = distance_squared_from_counts(xijk, target_fraction)
    end

    # in-stratum distance
    xijk = zeros(Int, n)
    for patient in axes(X, 2)
        instratum = true
        for i in eachindex(partition)
            if view(X, partition[i], patient) != view(Xcurrent, partition[i])
                instratum = false
                break
            end
        end
        if instratum
            xijk[W[patient]] += 1
        end
    end
    d_stratum = distance_squared_from_counts(xijk, target_fraction)

    # Gk is a weighted sum of the distances
    Gk = zeros(n)
    for k in 1:n
        d = vcat(d_overall[k], d_marginal[:, k], d_stratum[k])
        Gk[k] = sum(d .* weights)
    end
    return Gk
end

"""
    distance_squared_from_counts(counts, target_fraction)

Calculate the squared distance between the counts and the target fraction.
"""
function distance_squared_from_counts(counts, target_fraction)
    N = sum(counts)
    d = zeros(size(counts))
    for i in eachindex(counts)
        counts[i] += 1
        for j in eachindex(counts)
            d[i] += (counts[j] - target_fraction[j] * (N + 1))^2
        end
        counts[i] -= 1
    end
    return d
end

"""
    BiasedCoinPolicyLinear <: PolicyLinear
    BiasedCoinPolicyLinear(n, m, theta0, Sigma0, sample_std, predictive, prognostic, labeling=vcat(falses(m),trues(n*m)); p=vcat(0.5, 0.5*ones(n-1)/(n-1)), weights=vcat(0.0,ones(length(prognostic))/length(prognostic),0.0), target_fraction=ones(n)/n)
    BiasedCoinPolicyLinear(n, m, theta0, Sigma0, sample_std, FX::Union{CovariatesCopula,CovariatesIndependent}, labeling=vcat(falses(m),trues(n*m)); p=vcat(0.5, 0.5*ones(n-1)/(n-1)), weights=nothing, target_fraction=ones(n)/n)

A policy that allocates treatments according to a biased coin design and implements
using a linear model with a labeling.

See [Zhao, W., Ma, W., Wang, F., & Hu, F. (2022). Incorporating covariates information in adaptive clinical trials for precision medicine. Pharmaceutical Statistics, 21(1), 176-195.](https://doi.org/10.1002/pst.2160)
for the measure of the score. However, this policy allocates using a fixed `target_fraction`,
instead of a response-adaptive allocation. See [`RABC_OCBA_PolicyLinear`](@ref) for a response-adaptive
version of this policy.
"""
struct BiasedCoinPolicyLinear <: PolicyLinear
    model::BayesLinearRegression
    predictive::Vector{Int}
    prognostic::Vector{Vector{Int}}
    p::Vector{Float64}
    weights::Vector{Float64}
    target_fraction::Vector{Float64}
    function BiasedCoinPolicyLinear(model::BayesLinearRegression, predictive, prognostic, p, weights, target_fraction)
        all(predictive .∈ Ref(1:model.m)) || throw(ArgumentError("predictive must be a subset of 1:model.m"))
        for i in prognostic
            all(i .∈ Ref(1:model.m)) || throw(ArgumentError("each entry of prognostic must be a subset of 1:model.m"))
        end
        length(weights) == length(prognostic) + 2 || throw(ArgumentError("weights must have length equal to the number of prognostic plus 2"))
        sum(weights) ≈ 1 || @warn "weights do not add up to 1"
        target_fraction /= sum(target_fraction)
        sum(p) ≈ 1 || throw(ArgumentError("p must sum to 1"))
        all(diff(p) .<= 0) || @warn "p is not decreasing"
        new(model, predictive, prognostic, p, weights, target_fraction)
    end
end

function BiasedCoinPolicyLinear(n, m, theta0, Sigma0, sample_std, predictive, prognostic, labeling=vcat(falses(m), trues(n * m)); p=vcat(0.5, 0.5 * ones(n - 1) / (n - 1)), weights=vcat(0.0, ones(length(prognostic)) / length(prognostic), 0.0), target_fraction=ones(n) / n)
    BiasedCoinPolicyLinear(BayesLinearRegression(n, m, theta0, Sigma0, sample_std, labeling), predictive, prognostic, p, weights, target_fraction)
end

function BiasedCoinPolicyLinear(n, m, theta0, Sigma0, sample_std, FX::Union{CovariatesCopula,CovariatesIndependent}, labeling=vcat(falses(m), trues(n * m)); p=vcat(0.5, 0.5 * ones(n - 1) / (n - 1)), weights=nothing, target_fraction=ones(n) / n)
    length(FX) == m || throw(ArgumentError("length of FX must be equal to m"))
    predictive, prognostic = labeling2predprog(n, FX, labeling)
    if isnothing(weights)
        weights = vcat(0.0, ones(length(prognostic)) / length(prognostic), 0.0)
    end
    BiasedCoinPolicyLinear(BayesLinearRegression(n, m, theta0, Sigma0, sample_std, labeling), predictive, prognostic, p, weights, target_fraction)
end

function allocation(policy::BiasedCoinPolicyLinear, Xcurrent, W, X, Y, rng=Random.default_rng())
    index_subgroup = [view(X, policy.predictive, i) == view(Xcurrent, policy.predictive) for i in axes(X, 2)]
    W_subgroup = view(W, index_subgroup)
    X_subgroup = view(X, :, index_subgroup)
    score = biasedcoin_score(Xcurrent, W_subgroup, X_subgroup, policy.model.n, policy.prognostic, policy.weights, policy.target_fraction)
    treatment_order = sortperm(score, lt=(a, b) -> isless_ties(a, b, rng))
    return treatment_order[rand(rng, Categorical(policy.p))]
end

function isless_ties(a, b, rng)
    if isequal(a, b)
        return rand(rng, Bool)
    end
    return isless(a, b)
end

"""
    RABC_OCBA_PolicyLinear <: PolicyLinear
    RABC_OCBA_PolicyLinear(n, m, theta0, Sigma0, sample_std, predictive, prognostic, labeling=vcat(falses(m),trues(n*m)); p=vcat(0.5, 0.5*ones(n-1)/(n-1)), weights=vcat(0.0,ones(length(prognostic))/length(prognostic),0.0))
    RABC_OCBA_PolicyLinear(n, m, theta0, Sigma0, sample_std, FX::Union{CovariatesCopula,CovariatesIndependent}, labeling=vcat(falses(m),trues(n*m)); p=vcat(0.5, 0.5*ones(n-1)/(n-1)), weights=nothing)

Reasponse-Adaptive Biased Coin (RABC) using OCBA for the target fraction.
A policy that allocates treatments according to a response-adative biased coin design
and implements using a linear model with a labeling. The response-adaptive version
target fraction is updated using the OCBA method.

See [Zhao, W., Ma, W., Wang, F., & Hu, F. (2022). Incorporating covariates information in adaptive clinical trials for precision medicine. Pharmaceutical Statistics, 21(1), 176-195.](https://doi.org/10.1002/pst.2160)
for the measure of the score. 

See [BiasedCoinPolicyLinear](@ref) for a non-response-adaptive version of this policy.

See [OCBAPolicyLinear](@ref) for the OCBA policy without biased coin.

"""
struct RABC_OCBA_PolicyLinear <: PolicyLinear
    model::BayesLinearRegression
    predictive::Vector{Int}
    prognostic::Vector{Vector{Int}}
    p::Vector{Float64}
    weights::Vector{Float64}
    function RABC_OCBA_PolicyLinear(model::BayesLinearRegression, predictive, prognostic, p, weights)
        all(predictive .∈ Ref(1:model.m)) || throw(ArgumentError("predictive must be a subset of 1:model.m"))
        for i in prognostic
            all(i .∈ Ref(1:model.m)) || throw(ArgumentError("each entry of prognostic must be a subset of 1:model.m"))
        end
        length(weights) == length(prognostic) + 2 || throw(ArgumentError("weights must have length equal to the number of prognostic plus 2"))
        sum(weights) ≈ 1 || @warn "weights do not add up to 1"
        sum(p) ≈ 1 || throw(ArgumentError("p must sum to 1"))
        all(diff(p) .<= 0) || @warn "p is not decreasing"
        new(model, predictive, prognostic, p, weights)
    end
end

function RABC_OCBA_PolicyLinear(n, m, theta0, Sigma0, sample_std, predictive, prognostic, labeling=vcat(falses(m), trues(n * m)); p=vcat(0.5, 0.5 * ones(n - 1) / (n - 1)), weights=vcat(0.0, ones(length(prognostic)) / length(prognostic), 0.0))
    RABC_OCBA_PolicyLinear(BayesLinearRegression(n, m, theta0, Sigma0, sample_std, labeling), predictive, prognostic, p, weights)
end

function RABC_OCBA_PolicyLinear(n, m, theta0, Sigma0, sample_std, FX::Union{CovariatesCopula,CovariatesIndependent}, labeling=vcat(falses(m), trues(n * m)); p=vcat(0.5, 0.5 * ones(n - 1) / (n - 1)), weights=nothing)
    length(FX) == m || throw(ArgumentError("length of FX must be equal to m"))
    predictive, prognostic = labeling2predprog(n, FX, labeling)
    if isnothing(weights)
        weights = vcat(0.0, ones(length(prognostic)) / length(prognostic), 0.0)
    end
    RABC_OCBA_PolicyLinear(BayesLinearRegression(n, m, theta0, Sigma0, sample_std, labeling), predictive, prognostic, p, weights)
end

function allocation(policy::RABC_OCBA_PolicyLinear, Xcurrent, W, X, Y, rng=Random.default_rng())
    index_subgroup = [view(X, policy.predictive, i) == view(Xcurrent, policy.predictive) for i in axes(X, 2)]
    W_subgroup = view(W, index_subgroup)
    X_subgroup = view(X, :, index_subgroup)

    # Get the target fraction from OCBA
    p = [sum(W_subgroup .== w) for w in 1:policy.model.n]
    unobserved_treatments = findall(p .== 0)
    if length(unobserved_treatments) == 1
        return unobserved_treatments[1]
    elseif length(unobserved_treatments) > 1
        return rand(rng, unobserved_treatments)
    end
    p /= length(W_subgroup)
    means = [interact(iw, policy.model.n, Xcurrent, policy.model.labeling)' * policy.model.theta_t for iw in 1:policy.model.n]
    # Before computing the OCBA probabilities, we would like to identify the alternatives 
    # that are equivalent in the sense that they are perfectly correlated and 
    # have the same mean and variance. For simplicity, here we assume that arms
    # with same mean are equivalent. Those alternatives that are equivalent will
    # be passed as a single alternative to the OCBA algorithm and the probability 
    # of the OCBA algorithm will be evenly distributed among equivalent arms.
    means, m, c = unique_values_helper(means)
    vars = ones(length(means)) * policy.model.sample_std^2
    p_ocba = ocba_normal(means, vars)
    target_fraction = [p_ocba[m[i]] / c[i] for i in 1:policy.model.n]

    score = biasedcoin_score(Xcurrent, W_subgroup, X_subgroup, policy.model.n, policy.prognostic, policy.weights, target_fraction)
    treatment_order = sortperm(score, lt=(a, b) -> isless_ties(a, b, rng))
    return treatment_order[rand(rng, Categorical(policy.p))]
end