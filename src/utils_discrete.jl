# The functions in this file assume that FX is a `CovariatesIndependentFinite` or
# `CovariatesCopulaFinite` object with the default intercept=true and reduce_category=true.

"""
    X2g(X, FX::CovariatesGeneratorFinite)

Converts a design vector `X` with only discrete covariates into a group `g`,
where `g` is a unique integer assigned to each combination of the covariate values.
The reverse conversion can be achieved with [`g2X`](@ref).
"""
function X2g(X, FX::CovariatesGeneratorFinite)
    # Assumes FX has an intercept and reduce_category
    g = 1
    sumcat = 0
    prodcat = 1
    for i in length(FX.support):-1:1
        catn = length(FX.support[i])
        if FX.cat[i]
            Xtemp = X[(end-sumcat-catn+2):(end-sumcat)]
            gtemp = findall(x -> x == 1, Xtemp)
            if (length(gtemp) > 1)
                throw(DomainError("A category in X has more than one 1."))
            end
            ggtemp = isempty(gtemp) ? 0 : gtemp[1]
            g += ggtemp * prodcat
            sumcat += catn - 1
            prodcat *= catn
        else
            gtemp = findall(x -> x == X[end-sumcat], FX.support[i])
            g += (gtemp[1] - 1) * prodcat
            sumcat += 1
            prodcat *= catn
        end
    end
    return g
end


"""
    g2X(g, FX::CovariatesGeneratorFinite)

Converts a group `g` into a design vector `X`, where `g` is a unique integer
assigned to each combination of the covariate values. The reverse conversion can
be achieved with [`X2g`](@ref).
"""
function g2X(g, FX::CovariatesGeneratorFinite)
    # Assumes FX has an intercept and reduce_category
    catn = [length(i) for i in FX.support]
    splits = prod(catn)
    g > 0 || g <= splits || throw(BoundsError("The group `g=$g` is out of bounds [1,$splits]"))
    X = zeros(1 + sum([FX.cat[i] ? length(FX.support[i]) - 1 : 1 for i in eachindex(FX.support)]))
    X[1] = 1
    index = 1
    for i in eachindex(FX.support)
        splits = splits // catn[i]
        gcurrent = (g - 1) ÷ splits
        g = (g - 1) % splits + 1
        if FX.cat[i]
            if (gcurrent > 0)
                X[index+gcurrent] = 1
            end
            index = index + catn[i] - 1
        else
            X[index+1] = FX.support[i][gcurrent+1]
            index += 1
        end
    end
    return X
end

function total_groups(FX)
    return prod([length(s) for s in FX.support])
end

function index2treatment(i, gn)
    return (i - 1) ÷ gn + 1
end

function index2g(i, gn)
    return ((i - 1) % gn) + 1
end

function treatment_g2index(w, g, gn)
    return (w - 1) * gn + g
end

"""
    X2g_prior(theta0, Sigma0, FX::CovariatesGeneratorFinite, labeling, n)

Transform the prior for covariate values to the prior for groups.
"""
function X2g_prior(theta0, Sigma0, FX::CovariatesGeneratorFinite, labeling, n)
    gn = total_groups(FX)
    Xs = Matrix{Float64}(undef, length(FX), gn)
    for g in 1:gn
        Xs[:, g] = g2X(g, FX)
    end
    combs = gn * n
    theta0_disc = zeros(combs)
    for i in 1:combs
        w = index2treatment(i, gn)
        g = index2g(i, gn)
        theta0_disc[i] = interact(w, n, view(Xs, :, g), labeling)' * theta0
    end

    Sigma0_disc = zeros(combs, combs)
    for i in 1:combs
        wi = index2treatment(i, gn)
        gi = index2g(i, gn)
        for j in 1:i
            wj = index2treatment(j, gn)
            gj = index2g(j, gn)
            Sigma0_disc[i, j] = interact(wi, n, view(Xs, :, gi), labeling)' * Sigma0 * interact(wj, n, view(Xs, :, gj), labeling)
            Sigma0_disc[j, i] = Sigma0_disc[i, j]
        end
    end

    return theta0_disc, Sigma0_disc
end

"""
    X2g_probs(FX::CovariatesGeneratorFinite)

Obtain the probability of observing a group.
"""
function X2g_probs(FX::CovariatesIndependentFinite)
    catn = [length(s) for s in FX.support]
    p_disc = ones(prod(catn))
    for j in eachindex(p_disc)
        g = j
        splits = prod(catn)
        for i in eachindex(catn)
            splits = splits ÷ catn[i]
            gcurrent = (g - 1) ÷ splits
            g = (g - 1) % splits + 1
            p_disc[j] *= FX.probs[i][gcurrent+1]
        end
    end
    return p_disc
end

function X2g_probs(FX::CovariatesCopulaFinite)
    # This is a Monte Carlo approximation
    # An exact answer may be possible using the functionality of Copulas.jl
    catn = [length(s) for s in FX.support]
    p_disc = zeros(prod(catn))
    reps = length(p_disc) * 1000
    rng = Xoshiro(8765)
    for _ in 1:reps
        g = X2g(rand(rng, FX), FX)
        p_disc[g] += 1
    end
    p_disc ./= reps
    return p_disc
end
