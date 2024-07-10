"""
    LabelingSelector

Supetype for labeling selectors. Subtypes should implement the 
`labeling_selection` and `initialize!` functions.
"""
abstract type LabelingSelector end

"""
    labeling_selection(selection::LabelingSelector,W,X,Y,rng=Random.GLOBAL_RNG)

Select a labeling based on the vector of treatments `W`, covariates matrix `X`,
and vector of outcomes `Y`.
"""
function labeling_selection(selection::LabelingSelector,W,X,Y,rng=Random.GLOBAL_RNG)
    return vcat(falses(selection.m),trues(selection.n*selection.m))
end

"""
    initialize!(selection::LabelingSelector)

Initialize the labeling selector. This function is called before the start of the trial.
"""
function initialize!(selection::LabelingSelector)
    return
end

"""
    LassoCVLabelingSelector <: LabelingSelector
    LassoCVLabelingSelector(n,m,factor=0.0 [,labeling_base])

Labeling selector that uses Lasso with cross validation to select a labeling.
The `factor` parameter is used to select the best model from cross validation:
`factor == 0` selects the model with the lowest mean loss;
`factor > 0` selects the model with the lowest mean loss that is within `factor` standard erros of the best model. `labeling_base` is the initial labeling to use.
"""
struct LassoCVLabelingSelector <: LabelingSelector
    n::Int
    m::Int
    factor::Float64
    labeling_base::BitVector
    function LassoCVLabelingSelector(n,m,factor=0.0,labeling_base=falses(0))
        factor = abs(factor)
        if isempty(labeling_base)
            labeling_base = trues((n+1)*m)
        else
            (n+1)*m == length(labeling_base) || throw(ArgumentError("labeling_base is the wrong length: expected length=$((n+1)*m) vs. actual length=$(length(labeling_base))")) 
        end
        new(n,m,factor,labeling_base)
    end
end

function labeling_selection(selection::LassoCVLabelingSelector,W,X,Y,rng=Random.GLOBAL_RNG)
    labeling = copy(selection.labeling_base)
    labeling[1] = false # skip the intercept term because glmnet includes it automatically
    WX = interact(W,selection.n,X,labeling)
    cv = nothing
    try
        cv = glmnetcv(WX', Y, rng=rng)
    catch e
        # glmnetcv can fail if there are not enough observations/variation to fit the model
        # When this happens, we just return labelling_base
        # Warning: this fails silently. Bugs may be difficult to identify if glmnetcv is always failing.
        # println(e)
        labeling[1] = selection.labeling_base[1]
        return labeling
    end
    best_index = extract_bestindex_from_cv(selection, cv)
    betas = cv.path.betas[:,best_index]
    labeling[labeling] .= .!(betas .≈ 0.0)
    labeling[1] = abs(cv.path.a0[best_index]) > mean(abs.(betas))
    return labeling
end

function extract_bestindex_from_cv(selection::LassoCVLabelingSelector, cv)
    best_index = Base.argmin(cv.meanloss)
    if selection.factor != 0
        if selection.factor > 0
            candidate_indices = best_index-1:-1:1
        else
            candidate_indices = best_index+1:1:length(cv.meanloss)
        end
        for i in candidate_indices
            if cv.meanloss[i] > cv.meanloss[best_index] + selection.factor * cv.stdloss[best_index]
                best_index = i + (selection.factor > 0 ? 1 : -1)
                break
            end
        end
    end
    return best_index
end
