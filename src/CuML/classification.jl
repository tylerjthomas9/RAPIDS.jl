
# Model Structs
MMI.@mlj_model mutable struct LogisticRegression <: MMI.Probabilistic
    penalty::String = "l2"::(_ in ("none", "l1", "l2", "elasticnet"))
    tol::Float64 = 1e-4::(_ > 0)
    C::Float64 = 1.0::(_ > 0)
    fit_intercept::Bool = true
    class_weight = nothing
    max_iter::Int = 1000::(_ > 0)
    linesearch_max_iter::Int = 50::(_ > 0)
    solver::String = "qn"::(_ in ("qn", "lbfgs", "owl"))
    l1_ratio = nothing
    verbose::Bool = false
end

MMI.@mlj_model mutable struct MBSGDClassifier <: MMI.Probabilistic
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
    learning_rate::String =
        "constant"::(_ in ("adaptive", "constant", "invscaling", "optimal"))
    n_iter_no_change::Int = 5::(_ > 0)
    verbose::Bool = false
end

MMI.@mlj_model mutable struct RandomForestClassifier <: MMI.Probabilistic
    n_estimators::Int = 100::(_ > 0)
    split_criterion::Union{Int,String} = 0
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
    max_batch_size::Int = 4096::(_ > 0)
    random_state::Union{Nothing,Int} = nothing
    verbose::Bool = false
end

MMI.@mlj_model mutable struct SVC <: MMI.Probabilistic
    C::Float64 = 1.0::(_ > 0)
    kernel::String = "rbf"::(_ in ("linear", "poly", "rbf", "sigmoid"))
    degree::Int = 3::(_ > 0)
    gamma::Union{Int,String} = "scale"
    coef0::Float64 = 0.0
    tol::Float64 = 1e-3::(_ > 0)
    cache_size::Float64 = 1024.0::(_ > 0)
    class_weight::Union{String,Dict,Nothing} = nothing #TODO: check if dict works
    max_iter::Int = -1
    multiclass_strategy::String = "ovo"::(_ in ("ovo", "ovr"))
    nochange_steps::Int = 1000::(_ >= 0)
    probability::Bool = false
    random_state::Union{Nothing,Int} = nothing
    verbose::Bool = false
end

MMI.@mlj_model mutable struct LinearSVC <: MMI.Probabilistic
    penalty::String = "l2"::(_ in ("l1", "l2"))
    loss = "squared_hinge"::(_ in ("squared_hinge", "hinge"))
    fit_intercept::Bool = true
    penalized_intercept::Bool = true
    max_iter::Int = 1000::(_ > 0)
    linesearch_max_iter::Int = 100::(_ > 0)
    lbfgs_memory::Int = 5::(_ > 0)
    C::Float64 = 1.0::(_ >= 0)
    grad_tol::Float64 = 1e-4::(_ > 0)
    change_tol::Float64 = 1e-5::(_ > 0)
    tol::Union{Nothing,Float64} = nothing
    probability::Bool = false
    multi_class::String = "ovo"::(_ in ("ovo", "ovr"))
    verbose::Bool = false
end

MMI.@mlj_model mutable struct KNeighborsClassifier <: MMI.Probabilistic
    algorithm::String = "brute"::(_ in ("brute",))
    metric::String = "euclidean"
    weights::String = "uniform"::(_ in ("uniform",))
    verbose::Bool = false
end

# Multiple dispatch for initializing models
function model_init(mlj_model::LogisticRegression)
    return cuml.LogisticRegression(; mlj_to_kwargs(mlj_model)...)
end
model_init(mlj_model::MBSGDClassifier) = cuml.MBSGDClassifier(; mlj_to_kwargs(mlj_model)...)
function model_init(mlj_model::RandomForestClassifier)
    return cuml.RandomForestClassifier(; mlj_to_kwargs(mlj_model)...)
end
model_init(mlj_model::SVC) = cuml.svm.SVC(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::LinearSVC) = cuml.svm.LinearSVC(; mlj_to_kwargs(mlj_model)...)
function model_init(mlj_model::KNeighborsClassifier)
    return cuml.KNeighborsClassifier(; mlj_to_kwargs(mlj_model)...)
end

const CUML_CLASSIFICATION = Union{
    LogisticRegression,
    MBSGDClassifier,
    RandomForestClassifier,
    SVC,
    LinearSVC,
    KNeighborsClassifier,
}

# add metadata
MMI.load_path(::Type{<:LogisticRegression}) = "$PKG.LogisticRegression"
MMI.load_path(::Type{<:MBSGDClassifier}) = "$PKG.MBSGDClassifier"
MMI.load_path(::Type{<:RandomForestClassifier}) = "$PKG.RandomForestClassifier"
MMI.load_path(::Type{<:SVC}) = "$PKG.SVC"
MMI.load_path(::Type{<:LinearSVC}) = "$PKG.LinearSVC"
MMI.load_path(::Type{<:KNeighborsClassifier}) = "$PKG.KNeighborsClassifier"

function MMI.input_scitype(::Type{<:CUML_CLASSIFICATION})
    return Union{Table(MMI.Continuous),AbstractMatrix{<:MMI.Continuous}}
end
MMI.target_scitype(::Type{<:CUML_CLASSIFICATION}) = AbstractVector{<:Finite}

function MMI.docstring(::Type{<:LogisticRegression})
    return "cuML's LogisticRegression: https://docs.rapids.ai/api/cuml/stable/api.html#logistic-regression"
end
function MMI.docstring(::Type{<:MBSGDClassifier})
    return "cuML's MBSGDClassifier: https://docs.rapids.ai/api/cuml/stable/api.html#mini-batch-sgd-classifier"
end
function MMI.docstring(::Type{<:RandomForestClassifier})
    return "cuML's RandomForestClassifier: https://docs.rapids.ai/api/cuml/nightly/api.html#cuml.ensemble.RandomForestClassifier"
end
function MMI.docstring(::Type{<:SVC})
    return "cuML's SVC: https://docs.rapids.ai/api/cuml/nightly/api.html#cuml.svm.LinearSVC"
end
function MMI.docstring(::Type{<:LinearSVC})
    return "cuML's LinearSVC: https://docs.rapids.ai/api/cuml/nightly/api.html#cuml.svm.SVC"
end
function MMI.docstring(::Type{<:KNeighborsClassifier})
    return "cuML's KNeighborsClassifier: https://docs.rapids.ai/api/cuml/stable/api.html#nearest-neighbors-classification"
end

# fit methods
function MMI.fit(mlj_model::CUML_CLASSIFICATION, verbosity, X, y, w = nothing)
    X_numpy = to_numpy(X)
    y_numpy = to_numpy(y)

    # fit the model
    model = model_init(mlj_model)
    model.fit(X_numpy, y_numpy)
    fitresult = model

    # save result
    cache = nothing
    y_cat = MMI.categorical(y)
    classes_seen = filter(in(unique(y_cat)), MMI.classes(y_cat))
    report = (classes_seen = classes_seen, features = _feature_names(X))
    return (fitresult, cache, report)
end

# predict methods
function MMI.predict(mlj_model::Union{LogisticRegression, RandomForestClassifier, KNeighborsClassifier}, fitresult, Xnew)
    model = fitresult
    py_preds = model.predict_proba(to_numpy(Xnew))
    classes = pyconvert(Vector, model.classes_)
    preds = MMI.UnivariateFinite(classes, pyconvert(Array, py_preds), pool=missing)

    return preds
end

function MMI.predict(mlj_model::Union{SVC, LinearSVC}, fitresult, Xnew)
    model = fitresult
    classes = pyconvert(Vector, model.classes_)
    if pyconvert(Bool, model.probability)
        py_preds = model.predict_proba(to_numpy(Xnew))
        preds = MMI.UnivariateFinite(classes, pyconvert(Array, py_preds), pool=missing)
    else
        @warn "SVC was not trained with `probability=true`. Using class predictions."
        py_preds = model.predict(to_numpy(Xnew))
        preds = MMI.UnivariateFinite(classes, pyconvert(Array, py_preds), pool=missing, augment=true)
    end

    return preds
end


function MMI.predict(mlj_model::Union{MBSGDClassifier}, fitresult, Xnew)
    model = fitresult
    classes = pyconvert(Vector, model.classes_)
    py_preds = model.predict(to_numpy(Xnew))
    preds = MMI.UnivariateFinite(classes, pyconvert(Array, py_preds), pool=missing, augment=true)

    return preds
end

function MMI.predict_mean(mlj_model::CUML_CLASSIFICATION, fitresult, Xnew)
    model = fitresult
    py_preds = model.predict(to_numpy(Xnew))
    preds = MMI.categorical(pyconvert(Array, py_preds))

    return preds
end



# Classification metadata
MMI.metadata_pkg.(
    (
        LogisticRegression,
        MBSGDClassifier,
        RandomForestClassifier,
        SVC,
        LinearSVC,
        KNeighborsClassifier,
    ),
    name = "cuML Classification Methods",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,
)
