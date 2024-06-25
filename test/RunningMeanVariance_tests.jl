using ContextualBandits
using Statistics
using Test

@testset "RunningMeanVariance" begin
    agg = ContextualBandits.RunningMeanVariance()
    x = rand(1000)
    m = mean(x)
    v = var(x)
    for i in x
        ContextualBandits.update!(agg,i)
    end
    @test mean(agg) ≈ mean(x)
    @test var(agg) ≈ var(x)
    @test std(agg) ≈ std(x)
    @test ContextualBandits.mean_stderr(agg) ≈ std(x)/sqrt(1000)
    ContextualBandits.reset!(agg)
    @test mean(agg) == 0.0
    @test agg.S == 0.0
    @test agg.n == 0
    @test isnan(var(agg))
    @test isnan(std(agg))
    @test isnan(ContextualBandits.mean_stderr(agg))

    n = 1000
    breaks = Int.([floor(0.1*n),floor(0.7*n)])
    x = rand(n)
    agg1 = ContextualBandits.RunningMeanVariance{Float64}()
    agg2 = ContextualBandits.RunningMeanVariance{Float64}()
    agg3 = ContextualBandits.RunningMeanVariance{Float64}()
    for i in 1:breaks[1]
        ContextualBandits.update!(agg1,x[i])
    end
    for i in breaks[1]+1:breaks[2]
        ContextualBandits.update!(agg2,x[i])
    end
    for i in breaks[2]+1:n
        ContextualBandits.update!(agg3,x[i])
    end
    ContextualBandits.update!(agg1,agg2)
    ContextualBandits.update!(agg1,agg3)
    @test mean(agg1) ≈ mean(x)
    @test var(agg1) ≈ var(x)
    @test var(agg1,corrected=false) ≈ var(x,corrected = false)

end