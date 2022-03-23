
# Model hyperparameters

"""
RAPIDS AI Docs for Linear Regression: 
    https://docs.rapids.ai/api/cuml/stable/api.html#linear-regression

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = cuLinearRegression()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct cuLinearRegression <: MMI.Probabilistic
    handle = nothing
    algorithm::String = "eig"::(_ in ("svd", "eig", "qr", "svd-qr", "svd-jacobi"))
    fit_intercept::Bool = true
    normalize::Bool = false
    verbose::Bool = false
end


# Multiple dispatch for initializing models
model_init(mlj_model::cuLinearRegression) = cuml.LinearRegression(; mlj_to_kwargs(mlj_model)...)

# add metadata
MMI.metadata_model(cuLinearRegression,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's LinearRegression: https://docs.rapids.ai/api/cuml/stable/api.html#linear-regression",
	load_path    = "RAPIDS.cuLinearRegression"
)

const CUML_REGRESSION = Union{cuLinearRegression, }

function MMI.fit(mlj_model::CUML_REGRESSION, verbosity, X, y, w=nothing)
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
function MMI.predict(mlj_model::CUML_REGRESSION, fitresult, Xnew)
    model,  = fitresult
    py_preds = model.predict(prepare_x(Xnew))
    preds = pyconvert(Array, py_preds) 

    return preds
end
    