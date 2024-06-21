using Test

println("Running tests for utils.jl...")
include("utils_tests.jl")
println("Tests complete for utils.jl.")

println("Running tests for CovariatesGeneration.jl...")
include("CovariatesGeneration_tests.jl")
println("Tests complete for CovariatesGeneration.jl.")

println("Running tests for OutcomeModel folder...")
include("OutcomeModel_tests.jl")
println("Tests complete for OutcomeModel folder.")

println("Running tests for Policy folder...")
include("PolicyLabel_tests.jl")
println("Tests complete for Policy folder.")