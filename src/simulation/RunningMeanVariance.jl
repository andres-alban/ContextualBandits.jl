mutable struct RunningMeanVariance{T<:Number}
    M::T
    S::T
    n::Int
    RunningMeanVariance{T}() where {T<:Number} = new(zero(T), zero(T), 0)
end

RunningMeanVariance() = RunningMeanVariance{Float64}()
reset!(agg::RunningMeanVariance) = (agg.M = zero(agg.M); agg.S = zero(agg.S); agg.n = 0)


# https://www.johndcook.com/blog/standard_deviation/
# Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), 242-247
# See equation 1.5
function update!(agg::RunningMeanVariance,x)
    if isnan(x)
        return
    elseif agg.n == 0
        agg.n = 1
        agg.M = x
        agg.S = 0.0
    else
        agg.n += 1
        agg.S = agg.S + (agg.n-1)/agg.n * (x - agg.M)^2
        agg.M = agg.M + (x - agg.M) / agg.n
    end
    return
end

function update!(agg1::T,agg2::T) where T<:RunningMeanVariance
    agg1.S += agg2.S + (agg1.n * agg2.n / (agg1.n + agg2.n)) * (agg2.M - agg1.M)^2
    agg1.M += (agg2.n / (agg1.n + agg2.n)) * (agg2.M - agg1.M)
    agg1.n += agg2.n
end

Statistics.mean(agg::RunningMeanVariance) = agg.M
Statistics.var(agg::RunningMeanVariance; corrected=true) = agg.n > 1 ? agg.S / (agg.n - corrected) : NaN
Statistics.std(agg::RunningMeanVariance; corrected=true) = sqrt(var(agg; corrected=corrected))
mean_stderr(agg::RunningMeanVariance) = std(agg) / sqrt(agg.n)
