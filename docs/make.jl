using Documenter
using ContextualBandits

makedocs(sitename="ContextualBandits.jl",
    pages=[
        "index.md",
        "gettingstarted.md",
        "model.md",
        "simulation.md",
        "policies.md",
        "outcome_models.md",
        "covariates_generation.md"
    ]
)