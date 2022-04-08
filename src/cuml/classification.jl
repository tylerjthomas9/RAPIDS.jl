

# Model hyperparameters

"""
RAPIDS Docs for Logistic Regression: 
    https://docs.rapids.ai/api/cuml/stable/api.html#lgoistic-regression

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)
y = [repeat([0], 50)..., repeat([1], 50)...]

model = LogisticRegression()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
MLJModelInterface.@mlj_model mutable struct LogisticRegression <: MMI.Probabilistic
    handle = nothing
    penalty::String = "l2"::(_ in ("none", "l1", "l2", "elasticnet"))
    tol::Float64 = 1e-4::(_ > 0)
    C::Float64 = 1.0::(_ > 0)
    fit_intercept::Bool = true
    class_weight = nothing
    max_iter::Int = 1000::(_ > 0)
    linesearch_max_iter::Int = 50::(_ > 0)
    verbose::Bool = false
    l1_ratio = nothing
    solver::String = "qn"::(_ in ("qn", "lbfgs", "owl"))
end

"""
RAPIDS Docs for the MBSGD Classifier: 
    https://docs.rapids.ai/api/cuml/stable/api.html#mini-batch-sgd-classifier

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)
y = [repeat([0], 50)..., repeat([1], 50)...]

model = MBSGDClassifier()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
MLJModelInterface.@mlj_model mutable struct MBSGDClassifier <: MMI.Probabilistic
    handle = nothing
    loss::String = "squared_loss"::(_ in ("hinge", "log", "squared_loss"))
    penalty::String = "none"::(_ in ("none", "l1", "l2", "elasticnet"))
    alpha::Float64 = 0.0001::(_ >= 0)
    l1_ratio::Float64 = 0.15::(_ >= 0)
    batch_size::Int = 32::(_ > 0)
    fit_intercept::Bool = true
    epochs::Int = 1000::(_ > 0)
    tol::Float64 = 1e-3::(_ > 0)
    shuffle::Bool = true
    eta0::Float64 = 0.001::(_ > 0)
    power_t::Float64 = 0.5::(_ > 0)
    learning_rate::String = "constant"::(_ in ("adaptive", "constant", "invscaling", "optimal"))
    n_iter_no_change::Int = 5::(_ > 0)
    verbose::Bool = false
end

"""
RAPIDS Docs for the Nearest Neighbors Classifier: 
    https://docs.rapids.ai/api/cuml/stable/api.html#nearest-neighbors-classification

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)
y = [repeat([0], 50)..., repeat([1], 50)...]

model = KNeighborsClassifier()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
MLJModelInterface.@mlj_model mutable struct KNeighborsClassifier <: MMI.Probabilistic
    handle = nothing
    algorithm::String = "brute"::(_ in ("brute",))
    metric::String = "euclidean"
    weights::String = "uniform"::(_ in ("uniform",))
    verbose::Bool = false
end


# Multiple dispatch for initializing models
model_init(mlj_model::LogisticRegression) = cuml.LogisticRegression(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::MBSGDClassifier) = cuml.MBSGDClassifier(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::KNeighborsClassifier) = cuml.KNeighborsClassifier(; mlj_to_kwargs(mlj_model)...)

# add metadata
MMI.metadata_model(LogisticRegression,
    input_scitype   = AbstractMatrix,  
    output_scitype  = AbstractVector,  
    supports_weights = true,           
    descr = "cuML's LogisticRegression: https://docs.rapids.ai/api/cuml/stable/api.html#logistic-regression",
	load_path    = "RAPIDS.LogisticRegression"
)
MMI.metadata_model(MBSGDClassifier,
    input_scitype   = AbstractMatrix,  
    output_scitype  = AbstractVector,  
    supports_weights = false,                      
    descr = "cuML's MBSGDClassifier: https://docs.rapids.ai/api/cuml/stable/api.html#mini-batch-sgd-classifier",
	load_path    = "RAPIDS.MBSGDClassifier"
)
MMI.metadata_model(KNeighborsClassifier,
    input_scitype   = AbstractMatrix,  
    output_scitype  = AbstractVector,  
    supports_weights = false,                      
    descr = "cuML's KNeighborsClassifier: https://docs.rapids.ai/api/cuml/stable/api.html#nearest-neighbors-classification",
	load_path    = "RAPIDS.KNeighborsClassifier"
)

const CUML_CLASSIFICATION = Union{LogisticRegression, MBSGDClassifier, KNeighborsClassifier}

# fit methods
function MMI.fit(mlj_model::CUML_CLASSIFICATION, verbosity, X, y, w=nothing)
    # fit the model
    model = model_init(mlj_model)
    model.fit(prepare_input(X), prepare_input(y))
    fitresult = model

    # save result
    cache = nothing
    report = ()
    return (fitresult, cache, report)
end

# predict methods
function MMI.predict(mlj_model::CUML_CLASSIFICATION, fitresult, Xnew)
    model  = fitresult
    py_preds = model.predict(prepare_input(Xnew))
    preds = pyconvert(Array, py_preds) 

    return preds
end


# Classification metadata
MMI.metadata_pkg.((LogisticRegression, MBSGDClassifier, KNeighborsClassifier),
    name = "cuML Classification Methods",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url  = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,      # does it wrap around some other package?
)