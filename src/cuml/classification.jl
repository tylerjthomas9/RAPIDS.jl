

# Model hyperparameters

"""
RAPIDS Docs for Linear Regression: 
    https://docs.rapids.ai/api/cuml/stable/api.html#linear-regression

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = LinearRegression()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
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
    l2_ratio = nothing
    solver::String = "qn"::(_ in ("qn", "lbfgs", "owl"))
end

"""
RAPIDS Docs for the MBSGD Classifier: 
    https://docs.rapids.ai/api/cuml/stable/api.html#mini-batch-sgd-classifier

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = MBSGDClassifier()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct MBSGDClassifier <: MMI.Probabilistic
    handle = nothing
    loss::String = "squared_loss"::(_ in "hinge", "log", "squared_loss")
    penalty::String = "none"::(_ in ("none", "l1", "l2", "elasticnet"))
    alpha::Float64 = 0.0001::(_ >= 0)
end


# Multiple dispatch for initializing models
model_init(mlj_model::LogisticRegression) = cuml.LogisticRegression(; mlj_to_kwargs(mlj_model)...)


# add metadata
MMI.metadata_model(LogisticRegression,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's LogisticRegression: https://docs.rapids.ai/api/cuml/stable/api.html#logistic-regression",
	load_path    = "RAPIDS.LogisticRegression"
)

const CUML_CLASSIFICATION = Union{LogisticRegression, }

function MMI.fit(mlj_model::CUML_CLASSIFICATION, verbosity, X, y, w=nothing)
    # initialize model, prepare data
    model = model_init(mlj_model)

    # fit the model 
    # TODO: why do we have to specify numpy array?
    model.fit(prepare_x(X), prepare_y(y))
    fitresult = (model, )

    # save result
    cache = nothing
    report = (coef = pyconvert(Array, model.coef_), 
            intercept = pyconvert(Float64, model.intercept_)
    )
    return (fitresult, cache, report)
end


# predict methods
function MMI.predict(mlj_model::CUML_CLASSIFICATION, fitresult, Xnew)
    model,  = fitresult
    py_preds = model.predict(prepare_x(Xnew))
    preds = pyconvert(Array, py_preds) 

    return preds
end
    