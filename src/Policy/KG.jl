# Functions logEI, logSumExp, logEmaxAffine, affineBreakpoints, cKG, and iKg
# are Julia implementations of the Matlab functions at https://people.orie.cornell.edu/pfrazier/src.html
# based on the papers
# [Frazier, P., Powell, W., & Dayanik, S. (2009). The knowledge-gradient policy for correlated normal beliefs. INFORMS journal on Computing, 21(4), 599-613.](https://doi.org/10.1287/ijoc.1080.0314) and 
# [Frazier, P. I., Powell, W. B., & Dayanik, S. (2008). A knowledge-gradient policy for sequential information collection. SIAM Journal on Control and Optimization, 47(5), 2410-2439.](https://doi.org/10.1137/070693424)

function logEI(s)
    if s < -10
        return logpdf(Normal(),s) - log(s^2+1)
    else
        return log(s*cdf(Normal(),s) + pdf(Normal(),s))
    end
end

function logSumExp(x)
    xmax = maximum(x)
    if xmax == -Inf 
        return -Inf
    end
    return xmax + log(sum(exp.(x .- xmax)))
end

function logSumExp(x,y)
    if x == -Inf && y == -Inf
        return -Inf
    end
    if x >= y
        return x + log(1 + exp(y - x))
    else
        return y + log(1 + exp(x - y))
    end
end

"""
    logEmaxAffine(a,b)

Compute the logarithm of the function ``h(a,b) = E[max_i (a_i + b_i Z)]`` where ``Z`` is a standard normal random variable and ``a`` and ``b`` are vectors of the same length. 
The function is computed using the algorithm described in the paper
[Frazier, P., Powell, W., & Dayanik, S. (2009). The knowledge-gradient policy for correlated normal beliefs. INFORMS journal on Computing, 21(4), 599-613.](https://doi.org/10.1287/ijoc.1080.0314)
"""
function logEmaxAffine(a,b)
    # Rearrange a and b to get a_keep and b_keep
    # obtain breakpoints c
    b_keep, c_keep = affineBreakpoints(a,b)

    if length(b_keep) == 1
        return -Inf
    end
    return logSumExp(log.(diff(b_keep)) + logEI.(-abs.(c_keep)))
end

function affineBreakpoints(a,b)
    # sort a and b in ascending order of b solving ties in ascending order of a
    s = sortperm(a)
    a1 = a[s]
    b1 = b[s]
    s = sortperm(b1)
    a1 = a1[s]
    b1 = b1[s]
    # remove elements with same b leaving only the one with the largest a
    for i in 1:length(b1)-1
        if b1[i] == b1[i+1]
            b1[i] = -Inf
        end
    end
    b2 = b1[b1 .> -Inf]
    a2 = a1[b1 .> -Inf]

    M = length(a2)
    c = zeros(M+1)
    A = zeros(Int,M)

    c[1] = -Inf
    c[2] = Inf
    A[1] = 1
    Alen = 1
    for i in 1:M-1
        c[2+i] = Inf
        while true
            j = A[Alen]
            c[1+j] = (a2[j] - a2[i+1])/(b2[i+1] - b2[j])
            if Alen > 1 && c[1+j] <= c[1+A[Alen-1]]
                Alen = Alen - 1
            else
                break
            end
        end
        A[Alen+1] = i+1
        Alen = Alen + 1
    end
    keep = A[1:Alen]
    return b2[keep],c[keep[1:end-1] .+ 1]
end

"""
    cKG(theta,Sigma,sample_std)

Compute correlated Knowledge Gradient (cKG) indices following
[Frazier, P., Powell, W., & Dayanik, S. (2009). The knowledge-gradient policy for correlated normal beliefs. INFORMS journal on Computing, 21(4), 599-613.](https://doi.org/10.1287/ijoc.1080.0314)
"""
function cKG(theta,Sigma,sample_std)
    M = length(theta)
    logQ = Vector{Float64}(undef,M)
    for x in 1:M
        Sigma_x = view(Sigma,:,x)
        denominator = sqrt(Sigma_x[x] + sample_std[x]^2)
        if denominator == 0
            logQ[x] = -Inf
        else
            logQ[x] = logEmaxAffine(theta,Sigma_x ./ denominator)
        end
    end
    return logQ
end

function cKG(theta,Sigma,sample_std::Number)
    cKG(theta,Sigma,fill(sample_std,length(theta)))
end

"""
    iKG(theta,sigmasq,sample_std)

Compute Knowledge Gradient (KG) indices following
[Frazier, P. I., Powell, W. B., & Dayanik, S. (2008). A knowledge-gradient policy for sequential information collection. SIAM Journal on Control and Optimization, 47(5), 2410-2439.](https://doi.org/10.1137/070693424)
"""
function iKG(theta,sigmasq,sample_std)
    logQ = Vector{Float64}(undef,length(theta))
    for i in eachindex(theta)
        if sigmasq[i] <= 0
            logQ[i] = -Inf
        else
            sigmatilde = sigmasq[i] / sqrt(sigmasq[i] + sample_std[i]^2)
            delta = theta[i] - maximum(theta[eachindex(theta) .!= i])
            z = -abs(delta) / sigmatilde
            logQ[i] = log(sigmatilde) + logEI(z)
        end
    end
    return logQ
end

function iKG(theta,sigma,sample_std::Number)
    iKG(theta,sigma,fill(sample_std,length(theta)))
end