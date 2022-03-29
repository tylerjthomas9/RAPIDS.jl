
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
MLJModelInterface.@mlj_model mutable struct LinearRegression <: MMI.Probabilistic
    handle = nothing
    algorithm::String = "eig"::(_ in ("svd", "eig", "qr", "svd-qr", "svd-jacobi"))
    fit_intercept::Bool = true
    normalize::Bool = false
    verbose::Bool = false
end

"""
RAPIDS Docs for Ridge Regression: 
    https://docs.rapids.ai/api/cuml/stable/api.html#ridge-regression

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = Ridge()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct Ridge <: MMI.Probabilistic
    handle = nothing
    alpha::Float64 = 1.0::(_ > 0)
    solver::String = "eig"::(_ in ("svd", "eig", "cd"))
    fit_intercept::Bool = true
    normalize::Bool = false
    verbose::Bool = false
end


"""
RAPIDS Docs for Lasso Regression: 
    https://docs.rapids.ai/api/cuml/stable/api.html#lasso-regression

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = Lasso()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct Lasso <: MMI.Probabilistic
    handle = nothing
    alpha::Float64 = 1.0::(_ > 0)
    fit_intercept::Bool = true
    normalize::Bool = false
    max_iter::Int = 1000::(_ > 0)
    tol::Float64 = 1e-3::(_ > 0)
    selection::String = "cyclic"::(_ in ("cyclic", "random"))
    verbose::Bool = false
end

"""
RAPIDS Docs for ElasticNet: 
    https://docs.rapids.ai/api/cuml/stable/api.html#elasticnet-regression

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = ElasticNet()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct ElasticNet <: MMI.Probabilistic
    handle = nothing
    alpha::Float64 = 1.0::(_ > 0)
    l1_ratio::Float64 = 0.5::(_ > 0)
    fit_intercept::Bool = true
    normalize::Bool = false
    max_iter::Int = 1000::(_ > 0)
    tol::Float64 = 1e-3::(_ > 0)
    selection::String = "cyclic"::(_ in ("cyclic", "random"))
    verbose::Bool = false
end

"""
RAPIDS Docs for MBSGDRegressor: 
    https://docs.rapids.ai/api/cuml/stable/api.html#mini-batch-sgd-regressor
Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = MBSGDRegressor()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct MBSGDRegressor <: MMI.Probabilistic
    handle = nothing
    loss::String = "squared_loss"::(_ in ("squared_loss", ))
    penalty::String = "none"::(_ in ("none", "l1", "l2", "elasticnet"))
    alpha::Float64 = 0.0001::(_ >= 0)
    fit_intercept::Bool = true
    l1_ratio::Float64 = 0.15::(_ >= 0 && _ <= 1)
    batch_size::Int = 32::(_ > 0)
    epochs::Int = 1000::(_ > 0)
    tol::Float64 = 1e-3::(_ > 0)
    shuffle::Bool = true
    eta0::Float64 = 1e-3::(_ > 0)
    power_t::Float64 = 0.5::(_ > 0)
    learning_rate::String = "constant"::(_ in ("adaptive", "constant", "invscaling", "optimal"))
    n_iter_no_change::Int = 5::(_ > 0)
    verbose::Bool = false
end

"""
RAPIDS Docs for RandomForestRegressor: 
    https://docs.rapids.ai/api/cuml/stable/api.html#random-forest
Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = RandomForestRegressor()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct RandomForestRegressor <: MMI.Probabilistic
    handle = nothing
    n_estimators::Int = 100::(_ > 0)
    split_criterion = 2
    bootstrap::Bool = true
    max_samples::Float64 = 1.0::(_ > 0)
    max_depth::Int = 16::(_ > 0)
    max_leaves::Int = -1
    max_features = "auto"
    n_bins::Int = 128::(_ > 0)
    n_streams::Int = 4::(_ > 0)
    min_samples_leaf::Int = 1::(_ > 0)
    min_samples_split::Int = 2::(_ > 1)
    min_impurity_decrease::Float64 = 0.0::(_ >= 0)
    accuracy_metric::String = "r2"::(_ in ("median_ae", "mean_ae", "mse", "r2"))
    max_batch_size::Int = 4096::(_ > 0)
    random_state = nothing
    verbose::Bool = false
end

"""
RAPIDS Docs for Coordinate Descent: 
    https://docs.rapids.ai/api/cuml/stable/api.html#coordinate-descent

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = CD()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct CD <: MMI.Probabilistic
    handle = nothing
    loss::String = "squared_loss"::(_ in ("squared_loss", ))
    alpha::Float64 = 1.0::(_ > 0)
    l1_ratio::Float64 = 0.5::(_ > 0)
    fit_intercept::Bool = true
    max_iter::Int = 1000::(_ > 0)
    tol::Float64 = 1e-3::(_ > 0)
    shuffle::Bool = true
    verbose::Bool = false
end


"""
RAPIDS Docs for SVR: 
    https://docs.rapids.ai/api/cuml/stable/api.html#support-vector-machines

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = SVR()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct SVR <: MMI.Probabilistic
    handle = nothing
    C::Float64 = 1.0::(_ >= 0)
    kernel::String = "rbf"::(_ in ("linear", "poly", "rbf", "sigmoid"))
    degree::Int = 3::(_ > 0)
    gamma = "scale"
    coef0::Float64 = 0.0
    tol::Float64 = 1e-3::(_ > 0)
    epsilon::Float64 = 0.1::(_ >= 0)
    cache_size::Float64 = 1024.0::(_ > 0)
    max_iter::Int = -1
    nochange_steps::Int = 1000::(_ >= 0)
    verbose::Bool = false
end

"""
RAPIDS Docs for LinearSVR: 
    https://docs.rapids.ai/api/cuml/stable/api.html#support-vector-machines

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = SVLinearSVR()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct LinearSVR <: MMI.Probabilistic
    handle = nothing
    penalty::String = "l2"::(_ in ("l1", "l2"))
    loss = "epsilon_insensitive"::(_ in ("epsilon_insensitive", "squared_epsilon_insensitive"))
    fit_intercept::Bool = true
    penalized_intercept::Bool = true
    max_iter::Int = 1000::(_ > 0)
    linesearch_max_iter::Int = 100::(_ > 0)
    lbfgs_memory::Int = 5::(_ > 0)
    C::Float64 = 1.0::(_ >= 0)
    grad_tol::Float64 = 1e-4::(_ > 0)
    change_tol::Float64 = 1e-5::(_ > 0)
    tol = nothing
    epsilon::Float64 = 0.0::(_ >= 0)
    verbose::Bool = false
end


"""
RAPIDS Docs for Nearest Neighbors Regression: 
    https://docs.rapids.ai/api/cuml/stable/api.html#nearest-neighbors-regression

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)
y = rand(100)

model = KNeighborsRegressor()
mach = machine(model, x, y)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct KNeighborsRegressor <: MMI.Probabilistic
    handle = nothing
    algorithm::String = "brute"::(_ in ("brute",))
    metric::String = "euclidean"
    weights::String = "uniform"::(_ in ("uniform",))
    verbose::Bool = false
end







# Multiple dispatch for initializing models
model_init(mlj_model::LinearRegression) = cuml.LinearRegression(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::Ridge) = cuml.Ridge(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::Lasso) = cuml.Lasso(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::ElasticNet) = cuml.ElasticNet(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::MBSGDRegressor) = cuml.MBSGDRegressor(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::RandomForestRegressor) = cuml.RandomForestRegressor(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::CD) = cuml.CD(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::SVR) = cuml.svm.SVR(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::LinearSVR) = cuml.svm.LinearSVR(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::KNeighborsRegressor) = cuml.KNeighborsRegressor(; mlj_to_kwargs(mlj_model)...)




# add metadata
MMI.metadata_model(LinearRegression,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's LinearRegression: https://docs.rapids.ai/api/cuml/stable/api.html#linear-regression",
	load_path    = "RAPIDS.LinearRegression"
)
MMI.metadata_model(Ridge,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's Ridge: https://docs.rapids.ai/api/cuml/stable/api.html#ridge-regression",
	load_path    = "RAPIDS.Ridge"
)
MMI.metadata_model(Lasso,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's Lasso: https://docs.rapids.ai/api/cuml/stable/api.html#lassp-regression",
	load_path    = "RAPIDS.Lasso"
)
MMI.metadata_model(ElasticNet,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's ElasticNet: https://docs.rapids.ai/api/cuml/stable/api.html#elasticnet-regression",
	load_path    = "RAPIDS.ElasticNet"
)
MMI.metadata_model(MBSGDRegressor,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's MBSGDRegressor: https://docs.rapids.ai/api/cuml/stable/api.html#mini-batch-sgd-regressor",
	load_path    = "RAPIDS.MBSGDRegressor"
)
MMI.metadata_model(RandomForestRegressor,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's RandomForestRegressor: https://docs.rapids.ai/api/cuml/stable/api.html#random-forest",
	load_path    = "RAPIDS.RandomForestRegressor"
)
MMI.metadata_model(CD,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's Coordinate Descent: https://docs.rapids.ai/api/cuml/stable/api.html#coordinate-descent",
	load_path    = "RAPIDS.CD"
)
MMI.metadata_model(SVR,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's SVR: https://docs.rapids.ai/api/cuml/stable/api.html#support-vector-machines",
	load_path    = "RAPIDS.SVR"
)
MMI.metadata_model(LinearSVR,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's LinearSVR: https://docs.rapids.ai/api/cuml/stable/api.html#support-vector-machines",
	load_path    = "RAPIDS.LinearSVR"
)
MMI.metadata_model(KNeighborsRegressor,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's KNeighborsRegressor: https://docs.rapids.ai/api/cuml/stable/api.html#nearest-neighbors-regression",
	load_path    = "RAPIDS.KNeighborsRegressor"
)


const CUML_REGRESSION = Union{LinearRegression, 
                                Ridge,
                                Lasso,
                                ElasticNet,
                                MBSGDRegressor,
                                RandomForestRegressor,
                                CD,
                                SVR,
                                LinearSVR,
                                KNeighborsRegressor
                            }

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

function MMI.fit(mlj_model::RandomForestRegressor, verbosity, X, y, w=nothing)
    # initialize model, prepare data
    model = model_init(mlj_model)

    # fit the model 
    # TODO: why do we have to specify numpy array?
    model.fit(prepare_x(X), prepare_y(y))
    fitresult = (model, )

    # save result
    cache = nothing
    report = (n_features_in = pyconvert(Int, model.n_features_in_))
    return (fitresult, cache, report)
end

function MMI.fit(mlj_model::Union{SVR, KNeighborsRegressor}, verbosity, X, y, w=nothing)
    # initialize model, prepare data
    model = model_init(mlj_model)

    # fit the model 
    # TODO: why do we have to specify numpy array?
    model.fit(prepare_x(X), prepare_y(y))
    fitresult = (model, )

    # save result
    cache = nothing
    #TODO: Get params in report
    report = ()
    return (fitresult, cache, report)
end



# predict methods
function MMI.predict(mlj_model::CUML_REGRESSION, fitresult, Xnew)
    model,  = fitresult
    py_preds = model.predict(prepare_x(Xnew))
    preds = pyconvert(Array, py_preds) 

    return preds
end
    