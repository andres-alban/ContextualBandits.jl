using Test

println("Running tests for utils.jl...")
include("utils_tests.jl")
println("Tests complete for utils.jl.")

println("Running tests for CovariatesGeneration.jl...")
include("CovariatesGeneration_tests.jl")
println("Tests complete for CovariatesGeneration.jl.")

println("Running tests for utils_discrete.jl...")
include("utils_discrete_tests.jl")
println("Tests complete for utils_discrete.jl.")

println("Running tests for OutcomeModel folder...")
include("OutcomeModel_tests.jl")
println("Tests complete for OutcomeModel folder.")

println("Running tests for Policy folder...")
include("PolicyLinear_tests.jl")
include("robust_prior_linear_tests.jl")
include("KG_tests.jl")
include("fEVI_tests.jl")
println("Tests complete for Policy folder.")

println("Running tests for simulation folder...")
include("RunningMeanVariance_tests.jl")
include("simulation_tests.jl")
println("Tests complete for simulation folder.")