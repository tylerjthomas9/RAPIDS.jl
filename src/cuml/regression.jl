
# Model hyperparameters


MLJModelInterface.@mlj_model mutable struct LinearRegression <: MMI.Deterministic
    handle = nothing
    algorithm::String = "eig"::(_ in ("svd", "eig", "qr", "svd-qr", "svd-jacobi"))
    fit_intercept::Bool = true
    normalize::Bool = false
    verbose::Bool = false
end

MLJModelInterface.@mlj_model mutable struct Ridge <: MMI.Deterministic
    handle = nothing
    alpha::Float64 = 1.0::(_ > 0)
    solver::String = "eig"::(_ in ("svd", "eig", "cd"))
    fit_intercept::Bool = true
    normalize::Bool = false
    verbose::Bool = false
end

MLJModelInterface.@mlj_model mutable struct Lasso <: MMI.Deterministic
    handle = nothing
    alpha::Float64 = 1.0::(_ > 0)
    fit_intercept::Bool = true
    normalize::Bool = false
    max_iter::Int = 1000::(_ > 0)
    tol::Float64 = 1e-3::(_ > 0)
    selection::String = "cyclic"::(_ in ("cyclic", "random"))
    verbose::Bool = false
end

MLJModelInterface.@mlj_model mutable struct ElasticNet <: MMI.Deterministic
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

MLJModelInterface.@mlj_model mutable struct MBSGDRegressor <: MMI.Deterministic
    handle = nothing
    loss::String = "squared_loss"::(_ in ("squared_loss",))
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
    learning_rate::String =
        "constant"::(_ in ("adaptive", "constant", "invscaling", "optimal"))
    n_iter_no_change::Int = 5::(_ > 0)
    verbose::Bool = false
end

MLJModelInterface.@mlj_model mutable struct RandomForestRegressor <: MMI.Deterministic
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

MLJModelInterface.@mlj_model mutable struct CD <: MMI.Deterministic
    handle = nothing
    loss::String = "squared_loss"::(_ in ("squared_loss",))
    alpha::Float64 = 1.0::(_ > 0)
    l1_ratio::Float64 = 0.5::(_ > 0)
    fit_intercept::Bool = true
    max_iter::Int = 1000::(_ > 0)
    tol::Float64 = 1e-3::(_ > 0)
    shuffle::Bool = true
    verbose::Bool = false
end

MLJModelInterface.@mlj_model mutable struct SVR <: MMI.Deterministic
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

MLJModelInterface.@mlj_model mutable struct LinearSVR <: MMI.Deterministic
    handle = nothing
    penalty::String = "l2"::(_ in ("l1", "l2"))
    loss =
        "epsilon_insensitive"::(_ in ("epsilon_insensitive", "squared_epsilon_insensitive"))
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

MLJModelInterface.@mlj_model mutable struct KNeighborsRegressor <: MMI.Deterministic
    handle = nothing
    algorithm::String = "brute"::(_ in ("brute",))
    metric::String = "euclidean"
    weights::String = "uniform"::(_ in ("uniform",))
    verbose::Bool = false
end




# Multiple dispatch for initializing models
model_init(mlj_model::LinearRegression) =
    cuml.LinearRegression(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::Ridge) = cuml.Ridge(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::Lasso) = cuml.Lasso(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::ElasticNet) = cuml.ElasticNet(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::MBSGDRegressor) = cuml.MBSGDRegressor(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::RandomForestRegressor) =
    cuml.RandomForestRegressor(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::CD) = cuml.CD(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::SVR) = cuml.svm.SVR(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::LinearSVR) = cuml.svm.LinearSVR(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::KNeighborsRegressor) =
    cuml.KNeighborsRegressor(; mlj_to_kwargs(mlj_model)...)

const CUML_REGRESSION = Union{
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    MBSGDRegressor,
    RandomForestRegressor,
    CD,
    SVR,
    LinearSVR,
    KNeighborsRegressor,
}

# add metadata
MMI.load_path(::Type{<:LinearRegression}) = "$PKG.LinearRegression"
MMI.load_path(::Type{<:Ridge}) = "$PKG.Ridge"
MMI.load_path(::Type{<:Lasso}) = "$PKG.Lasso"
MMI.load_path(::Type{<:ElasticNet}) = "$PKG.ElasticNet"
MMI.load_path(::Type{<:MBSGDRegressor}) = "$PKG.MBSGDRegressor"
MMI.load_path(::Type{<:RandomForestRegressor}) = "$PKG.RandomForestRegressor"
MMI.load_path(::Type{<:CD}) = "$PKG.CD"
MMI.load_path(::Type{<:SVR}) = "$PKG.SVR"
MMI.load_path(::Type{<:LinearSVR}) = "$PKG.LinearSVR"
MMI.load_path(::Type{<:KNeighborsRegressor}) = "$PKG.KNeighborsRegressor"

MMI.input_scitype(::Type{<:CUML_REGRESSION}) = Union{Table(Continuous), AbstractMatrix{<:Continuous}}
MMI.target_scitype(::Type{<:CUML_REGRESSION}) = AbstractVector{<:Continuous}

MMI.docstring(::Type{<:LinearRegression}) =
    "cuML's LinearRegression: https://docs.rapids.ai/api/cuml/stable/api.html#linear-regression"
MMI.docstring(::Type{<:Ridge}) =
    "cuML's Ridge: https://docs.rapids.ai/api/cuml/stable/api.html#ridge-regression"
MMI.docstring(::Type{<:Lasso}) =
    "cuML's Lasso: https://docs.rapids.ai/api/cuml/stable/api.html#lasso-regression"
MMI.docstring(::Type{<:ElasticNet}) =
    "cuML's ElasticNet: https://docs.rapids.ai/api/cuml/stable/api.html#elasticnet-regression"
MMI.docstring(::Type{<:MBSGDRegressor}) =
    "cuML's MBSGDRegressor: https://docs.rapids.ai/api/cuml/stable/api.html#mini-batch-sgd-regressor"
MMI.docstring(::Type{<:RandomForestRegressor}) =
    "cuML's RandomForestRegressor: https://docs.rapids.ai/api/cuml/stable/api.html#random-forest"
MMI.docstring(::Type{<:CD}) =
    "cuML's CD: https://docs.rapids.ai/api/cuml/stable/api.htmll#coordinate-descent"
MMI.docstring(::Type{<:SVR}) =
    "cuML's SVR: https://docs.rapids.ai/api/cuml/stable/api.html#support-vector-machines"
MMI.docstring(::Type{<:LinearSVR}) =
    "cuML's LinearSVR: https://docs.rapids.ai/api/cuml/stable/api.html#support-vector-machines"
MMI.docstring(::Type{<:KNeighborsRegressor}) =
    "cuML's KNeighborsRegressor: https://docs.rapids.ai/api/cuml/stable/api.html#nearest-neighbors-regression"


# fit methods
function MMI.fit(mlj_model::CUML_REGRESSION, verbosity, X, y, w = nothing)
    schema = Tables.schema(X)
    X_numpy = prepare_input(X)
    y_numpy  = prepare_input(y)

    if schema === nothing
        features = [Symbol("x$j") for j in 1:size(X, 2)]
    else
        features = schema.names |> collect
    end

    # fit the model
    model = model_init(mlj_model)
    model.fit(X_numpy, y_numpy)
    fitresult = model

    # save result
    cache = nothing
    report = (features=features)
    return (fitresult, cache, report)
end

# predict methods
function MMI.predict(mlj_model::CUML_REGRESSION, fitresult, Xnew)
    model = fitresult
    py_preds = model.predict(prepare_input(Xnew))
    preds = pyconvert(Array, py_preds)

    return preds
end

# Regression metadata
MMI.metadata_pkg.(
    (
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
        MBSGDRegressor,
        RandomForestRegressor,
        CD,
        SVR,
        LinearSVR,
        KNeighborsRegressor,
    ),
    name = "cuML Regression Methods",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,      # does it wrap around some other package?
)

# docstrings

"""
$(MMI.doc_header(LinearRegression))

`LinearRegression`  is a wrapper for the RAPIDS Linear Regression.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`
- `y`: is an `AbstractVector` continuous target.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters
- `algorithm="eig"`: 
    - `eig`: use an eigendecomposition of the covariance matrix.
    - `qr`: use QR decomposition algorithm and solve `Rx = Q^T y`
    - `svd`: alias for svd-jacobi.
    - `svd-qr`: compute SVD decomposition using QR algorithm.
    - `svd-jacobi`: compute SVD decomposition using Jacobi iterations.
- `fit_intercept=true`: If true, the model tries to correct for the global mean of y. 
                        If false, the model expects that you have centered the data.
- `normalize=true`: This parameter is ignored when fit_intercept is set to false.
                    If true, the predictors in X will be normalized by dividing by the column-wise standard deviation. 
                    If false, no scaling will be done. 
- `verbose=false`: Sets logging level.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are class assignments. 

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:
- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)

# Examples
```
using RAPIDS
using MLJ

X = rand(100, 5)
y = rand(100)

model = LinearRegression()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
LinearRegression


"""
$(MMI.doc_header(RIDGE))

`LinearRegression`  is a wrapper for the RAPIDS RIDGE Regression.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`
- `y`: is an `AbstractVector` continuous target.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters
- `alpha=1.0`: Regularization strength - must be a positive float. Larger values specify stronger regularization.
- `solver="eig"`: 
    - `cd`: use coordinate descent. Very fast and is suitable for large problems.
    - `eig`: use an eigendecomposition of the covariance matrix.
    - `svd`: alias for svd-jacobi. Slower, but guaranteed to be stable.
- `fit_intercept=true`: If true, the model tries to correct for the global mean of y. 
                        If false, the model expects that you have centered the data.
- `normalize=true`: This parameter is ignored when fit_intercept is set to false.
                    If true, the predictors in X will be normalized by dividing by the column-wise standard deviation. 
                    If false, no scaling will be done. 
- `verbose=false`: Sets logging level.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are class assignments. 

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:
- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)

# Examples
```
using RAPIDS
using MLJ

X = rand(100, 5)
y = rand(100)

model = RIDGE()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
RIDGE


"""
$(MMI.doc_header(LASSO))

`LinearRegression`  is a wrapper for the RAPIDS LASSO Regression.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`
- `y`: is an `AbstractVector` continuous target.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters
- `alpha=1.0`: Constant that multiplies the L1 term. alpha = 0 is equivalent to an ordinary least square.
- `tol=1e-4': Tolerance for stopping criteria. 
- `max_iter=1000`: Maximum number of iterations taken for the solvers to converge.
- `solver="cd"`: 
    - `cd`: Coordinate descent.
    - `qn`: quasi-newton. You may find the alternative ‘qn’ algorithm is faster when the number of features is sufficiently large, but the sample size is small.
- `fit_intercept=true`: If true, the model tries to correct for the global mean of y. 
                        If false, the model expects that you have centered the data.
- `normalize=true`: This parameter is ignored when fit_intercept is set to false.
                    If true, the predictors in X will be normalized by dividing by the column-wise standard deviation. 
                    If false, no scaling will be done. 
- `selection="cyclic"`: 
    - `cyclic`: loop over features to update coefficients.
    - `random`: a random coefficient is updated every iteration.
    a random coefficient is updated every iteration rather than looping over features sequentially by default.
- `verbose=false`: Sets logging level.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are class assignments. 

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:
- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)

# Examples
```
using RAPIDS
using MLJ

X = rand(100, 5)
y = rand(100)

model = LASSO()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
LASSO

