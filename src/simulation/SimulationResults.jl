"""
    SimulationResultsAggregator

Supertype of objects that aggregate simulation results used in [simulation_stochastic](@ref).
"""
abstract type SimulationResultsAggregator end

"""
    StandardResultsAggregator

Aggregator for simulation results that computes the mean and standard deviation of the results.
"""
struct StandardResultsAggregator <: SimulationResultsAggregator
    agg
    names::Vector{String}
    function StandardResultsAggregator(output, names)
        @assert length(output) == length(names)
        agg = Tuple([RunningMeanVariance() for _ in x] for x in output)
        new(agg, names)
    end
end

function update!(sra::StandardResultsAggregator, output)
    for (i, x) in enumerate(output)
        for (j, y) in enumerate(x)
            update!(sra.agg[i][j], y)
        end
    end
end

function update!(sra::StandardResultsAggregator, sra2::StandardResultsAggregator)
    update!(sra, sra2.agg)
end

function asdict(sra::StandardResultsAggregator)
    out = Dict{String,Any}()
    for (i, name) in enumerate(sra.names)
        out[name] = Dict(
            "mean" => mean.(sra.agg[i]), "std" => std.(sra.agg[i]), "n" => [j.n for j in sra.agg[i]]
        )
    end
    return out
end
