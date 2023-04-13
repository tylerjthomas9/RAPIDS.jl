
# Model hyperparameters

MMI.@mlj_model mutable struct LinearRegression <: MMI.Deterministic
    algorithm::String = "eig"::(_ in ("svd", "eig", "qr", "svd-qr", "svd-jacobi"))
    fit_intercept::Bool = true
    normalize::Bool = false
    verbose::Bool = false
end

MMI.@mlj_model mutable struct Ridge <: MMI.Deterministic
    alpha::Float64 = 1.0::(_ > 0)
    solver::String = "eig"::(_ in ("svd", "eig", "cd"))
    fit_intercept::Bool = true
    normalize::Bool = false
    verbose::Bool = false
end

MMI.@mlj_model mutable struct Lasso <: MMI.Deterministic
    alpha::Float64 = 1.0::(_ > 0)
    fit_intercept::Bool = true
    normalize::Bool = false
    max_iter::Int = 1000::(_ > 0)
    tol::Float64 = 1e-3::(_ > 0)
    selection::String = "cyclic"::(_ in ("cyclic", "random"))
    verbose::Bool = false
end

MMI.@mlj_model mutable struct ElasticNet <: MMI.Deterministic
    alpha::Float64 = 1.0::(_ > 0)
    l1_ratio::Float64 = 0.5::(_ > 0)
    fit_intercept::Bool = true
    normalize::Bool = false
    max_iter::Int = 1000::(_ > 0)
    tol::Float64 = 1e-3::(_ > 0)
    solver::String = "cd"::(_ in ("cd", "qn"))
    selection::String = "cyclic"::(_ in ("cyclic", "random"))
    verbose::Bool = false
end

MMI.@mlj_model mutable struct MBSGDRegressor <: MMI.Deterministic
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

MMI.@mlj_model mutable struct RandomForestRegressor <: MMI.Deterministic
    n_estimators::Int = 100::(_ > 0)
    split_criterion::Union{Int,String} = 2
    bootstrap::Bool = true
    max_samples::Float64 = 1.0::(_ > 0)
    max_depth::Int = 16::(_ > 0)
    max_leaves::Int = -1
    max_features::Union{Int,Float64,String} = "auto"
    n_bins::Int = 128::(_ > 0)
    n_streams::Int = 4::(_ > 0)
    min_samples_leaf::Union{Int,Float64} = 1
    min_samples_split::Union{Int,Float64} = 2
    min_impurity_decrease::Float64 = 0.0::(_ >= 0)
    accuracy_metric::String = "r2"::(_ in ("median_ae", "mean_ae", "mse", "r2"))
    max_batch_size::Int = 4096::(_ > 0)
    random_state::Union{Nothing,Int} = nothing
    verbose::Bool = false
end

MMI.@mlj_model mutable struct CD <: MMI.Deterministic
    loss::String = "squared_loss"::(_ in ("squared_loss",))
    alpha::Float64 = 0.0001::(_ > 0)
    l1_ratio::Float64 = 0.15::(_ > 0)
    normalize::Bool = true
    fit_intercept::Bool = true
    max_iter::Int = 1000::(_ > 0)
    tol::Float64 = 1e-3::(_ > 0)
    shuffle::Bool = true
    verbose::Bool = false
end

MMI.@mlj_model mutable struct SVR <: MMI.Deterministic
    C::Float64 = 1.0::(_ > 0)
    kernel::String = "rbf"::(_ in ("linear", "poly", "rbf", "sigmoid"))
    degree::Int = 3::(_ > 0)
    gamma::Union{Int,String} = "scale"
    coef0::Float64 = 0.0
    tol::Float64 = 1e-3::(_ > 0)
    epsilon::Float64 = 0.1::(_ >= 0)
    cache_size::Float64 = 1024.0::(_ > 0)
    max_iter::Int = -1
    nochange_steps::Int = 1000::(_ >= 0)
    verbose::Bool = false
end

MMI.@mlj_model mutable struct LinearSVR <: MMI.Deterministic
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
    tol::Union{Nothing,Float64} = nothing
    epsilon::Float64 = 0.0::(_ >= 0)
    verbose::Bool = false
end

MMI.@mlj_model mutable struct KNeighborsRegressor <: MMI.Deterministic
    n_neighbors::Int = 5::(_ > 0)
    algorithm::String = "auto"::(_ in ("auto", "brute", "rbc", "ivfflat", "ivfpq", "ivfsq"))
    metric::String = "euclidean"
    weights::String = "uniform"::(_ in ("uniform",))
    verbose::Bool = false
end

# Multiple dispatch for initializing models
function model_init(mlj_model::LinearRegression)
    return cuml.LinearRegression(; mlj_to_kwargs(mlj_model)...)
end
model_init(mlj_model::Ridge) = cuml.Ridge(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::Lasso) = cuml.Lasso(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::ElasticNet) = cuml.ElasticNet(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::MBSGDRegressor) = cuml.MBSGDRegressor(; mlj_to_kwargs(mlj_model)...)
function model_init(mlj_model::RandomForestRegressor)
    return cuml.RandomForestRegressor(; mlj_to_kwargs(mlj_model)...)
end
model_init(mlj_model::CD) = cuml.CD(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::SVR) = cuml.svm.SVR(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::LinearSVR) = cuml.svm.LinearSVR(; mlj_to_kwargs(mlj_model)...)
function model_init(mlj_model::KNeighborsRegressor)
    return cuml.KNeighborsRegressor(; mlj_to_kwargs(mlj_model)...)
end

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
MMI.load_path(::Type{<:LinearRegression}) = "$PKG.CuML.LinearRegression"
MMI.load_path(::Type{<:Ridge}) = "$PKG.CuML.Ridge"
MMI.load_path(::Type{<:Lasso}) = "$PKG.CuML.Lasso"
MMI.load_path(::Type{<:ElasticNet}) = "$PKG.CuML.ElasticNet"
MMI.load_path(::Type{<:MBSGDRegressor}) = "$PKG.CuML.MBSGDRegressor"
MMI.load_path(::Type{<:RandomForestRegressor}) = "$PKG.CuML.RandomForestRegressor"
MMI.load_path(::Type{<:CD}) = "$PKG.CuML.CD"
MMI.load_path(::Type{<:SVR}) = "$PKG.CuML.SVR"
MMI.load_path(::Type{<:LinearSVR}) = "$PKG.CuML.LinearSVR"
MMI.load_path(::Type{<:KNeighborsRegressor}) = "$PKG.CuML.KNeighborsRegressor"

function MMI.input_scitype(::Type{<:CUML_REGRESSION})
    return Union{
        MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor, MMI.Multiclass),
        AbstractMatrix{MMI.Continuous},
    }
end
MMI.target_scitype(::Type{<:CUML_REGRESSION}) = AbstractVector{<:MMI.Continuous}

function MMI.docstring(::Type{<:LinearRegression})
    return "cuML's LinearRegression: https://docs.rapids.ai/api/cuml/stable/api.html#linear-regression"
end
function MMI.docstring(::Type{<:Ridge})
    return "cuML's Ridge: https://docs.rapids.ai/api/cuml/stable/api.html#ridge-regression"
end
function MMI.docstring(::Type{<:Lasso})
    return "cuML's Lasso: https://docs.rapids.ai/api/cuml/stable/api.html#lasso-regression"
end
function MMI.docstring(::Type{<:ElasticNet})
    return "cuML's ElasticNet: https://docs.rapids.ai/api/cuml/stable/api.html#elasticnet-regression"
end
function MMI.docstring(::Type{<:MBSGDRegressor})
    return "cuML's MBSGDRegressor: https://docs.rapids.ai/api/cuml/stable/api.html#mini-batch-sgd-regressor"
end
function MMI.docstring(::Type{<:RandomForestRegressor})
    return "cuML's RandomForestRegressor: https://docs.rapids.ai/api/cuml/stable/api.html#random-forest"
end
function MMI.docstring(::Type{<:CD})
    return "cuML's CD: https://docs.rapids.ai/api/cuml/stable/api.htmll#coordinate-descent"
end
function MMI.docstring(::Type{<:SVR})
    return "cuML's SVR: https://docs.rapids.ai/api/cuml/stable/api.html#support-vector-machines"
end
function MMI.docstring(::Type{<:LinearSVR})
    return "cuML's LinearSVR: https://docs.rapids.ai/api/cuml/stable/api.html#support-vector-machines"
end
function MMI.docstring(::Type{<:KNeighborsRegressor})
    return "cuML's KNeighborsRegressor: https://docs.rapids.ai/api/cuml/stable/api.html#nearest-neighbors-regression"
end

# fit methods
function MMI.fit(mlj_model::CUML_REGRESSION, verbosity, X, y, w=nothing)
    # fit the model
    model = model_init(mlj_model)
    model.fit(X, y)
    fitresult = model

    # save result
    cache = nothing
    report = (features = _feature_names(X))
    return (fitresult, cache, report)
end

# predict methods
function MMI.predict(mlj_model::CUML_REGRESSION, fitresult, Xnew)
    model = fitresult
    py_preds = model.predict(Xnew)
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
    name="cuML Regression Methods",
    uuid="2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url="https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia=false,          # is it written entirely in Julia?
    license="MIT",        # your package license
    is_wrapper=true,
)
