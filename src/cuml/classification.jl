
# Model Structs
MLJModelInterface.@mlj_model mutable struct LogisticRegression <: MMI.Probabilistic
    handle = nothing
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
    learning_rate::String =
        "constant"::(_ in ("adaptive", "constant", "invscaling", "optimal"))
    n_iter_no_change::Int = 5::(_ > 0)
    verbose::Bool = false
end

MLJModelInterface.@mlj_model mutable struct KNeighborsClassifier <: MMI.Probabilistic
    handle = nothing
    algorithm::String = "brute"::(_ in ("brute",))
    metric::String = "euclidean"
    weights::String = "uniform"::(_ in ("uniform",))
    verbose::Bool = false
end


# Multiple dispatch for initializing models
model_init(mlj_model::LogisticRegression) =
    cuml.LogisticRegression(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::MBSGDClassifier) = cuml.MBSGDClassifier(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::KNeighborsClassifier) =
    cuml.KNeighborsClassifier(; mlj_to_kwargs(mlj_model)...)

const CUML_CLASSIFICATION = Union{LogisticRegression,MBSGDClassifier,KNeighborsClassifier}

# add metadata
MMI.load_path(::Type{<:LogisticRegression}) = "$PKG.LogisticRegression"
MMI.load_path(::Type{<:MBSGDClassifier}) = "$PKG.MBSGDClassifier"
MMI.load_path(::Type{<:KNeighborsClassifier}) = "$PKG.KNeighborsClassifier"


MMI.input_scitype(::Type{<:CUML_CLASSIFICATION}) = Tuple{Union{Table{<:AbstractVector{<:Continuous}}, AbstractMatrix{<:Continuous}}, AbstractVector{<:Finite}}
MMI.target_scitype(::Type{<:CUML_CLASSIFICATION}) = AbstractVector{<:Finite}

MMI.docstring(::Type{<:LogisticRegression}) =
    "cuML's LogisticRegression: https://docs.rapids.ai/api/cuml/stable/api.html#logistic-regression"
MMI.docstring(::Type{<:MBSGDClassifier}) =
    "cuML's MBSGDClassifier: https://docs.rapids.ai/api/cuml/stable/api.html#mini-batch-sgd-classifier"
MMI.docstring(::Type{<:KNeighborsClassifier}) =
    "cuML's KNeighborsClassifier: https://docs.rapids.ai/api/cuml/stable/api.html#nearest-neighbors-classification"

# fit methods
function MMI.fit(mlj_model::CUML_CLASSIFICATION, verbosity, X, y, w = nothing)
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
    y_cat = y |> MMI.categorical
    classes_seen = filter(in(unique(y_cat)), MMI.classes(y_cat))
    report = (classes_seen=classes_seen,
                features=features)
    return (fitresult, cache, report)
end

# predict methods
function MMI.predict(mlj_model::CUML_CLASSIFICATION, fitresult, Xnew)
    model = fitresult
    py_preds = model.predict(prepare_input(Xnew))
    preds = pyconvert(Array, py_preds) |> MMI.categorical
 
    return preds
end


# Classification metadata
MMI.metadata_pkg.(
    (LogisticRegression, MBSGDClassifier, KNeighborsClassifier),
    name = "cuML Classification Methods",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,      # does it wrap around some other package?
)

# docstrings
# TODO: add Table/DataFrame examples

"""
$(MMI.doc_header(LogisticRegression))

`LogisticRegression`  is a wrapper for the RAPIDS Logistic Regression.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

- `y`: is the target, which can be any `AbstractVector` whose element
    scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
    with `scitype(y)`
Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters
- `penalty="l2"`: Normalization/penalty function ("none", "l1", "l2", "elasticnet").
    - `none`: the L-BFGS solver will be used
    - `l1`: The L1 penalty is best when there are only a few useful features (sparse), and you
            want to zero out non-important features. The L-BFGS solver will be used.
    - `l2`: The L2 penalty is best when you have a lot of important features, especially if they
            are correlated.The L-BFGS solver will be used.
    - `elasticnet`: A combination of the L1 and L2 penalties. The OWL-QN solver will be used if
                    `l1_ratio>0`, otherwise the L-BFGS solver will be used.
- `tol=1e-4': Tolerance for stopping criteria. 
- `C=1.0`: Inverse of regularization strength.
- `fit_intercept=true`: If True, the model tries to correct for the global mean of y. 
                        If False, the model expects that you have centered the data.
- `class_weight="balanced"`: Dictionary or `"balanced"`.
- `max_iter=1000`: Maximum number of iterations taken for the solvers to converge.
- `linesearch_max_iter=50`: Max number of linesearch iterations per outer iteration used in 
                            the lbfgs and owl QN solvers.
- `solver="qn"`: Algorithm to use in the optimization problem. Currently only `qn` is
                 supported, which automatically selects either `L-BFGS `or `OWL-QN`
- `l1_ratio=nothing`: The Elastic-Net mixing parameter. 
- `verbose=false`: Sets logging level.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are class assignments. 

- `predict_proba(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are probabilistic, but uncalibrated.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:

- `classes_seen`: list of target classes actually observed in training

- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)

# Examples
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
LogisticRegression



"""
$(MMI.doc_header(MBSGDClassifier))

`MBSGDClassifier`  is a wrapper for the RAPIDS Mini Batch SGD Classifier.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

- `y`: is the target, which can be any `AbstractVector` whose element
    scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
    with `scitype(y)`
Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `loss="squared_loss"`: Loss function ("hinge", "log", "squared_loss").
    - `hinge`: Linear SVM
    - `log`: Logistic regression
    - `squared_loss`: Linear regression
- `penalty="none"`: Normalization/penalty function ("none", "l1", "l2", "elasticnet").
    - `none`: the L-BFGS solver will be used
    - `l1`: The L1 penalty is best when there are only a few useful features (sparse), and you
            want to zero out non-important features. The L-BFGS solver will be used.
    - `l2`: The L2 penalty is best when you have a lot of important features, especially if they
            are correlated.The L-BFGS solver will be used.
    - `elasticnet`: A combination of the L1 and L2 penalties. The OWL-QN solver will be used if
                    `l1_ratio>0`, otherwise the L-BFGS solver will be used.
- `alpha=1e-4`: The constant value which decides the degree of regularization.
- `l1_ratio=nothing`: The Elastic-Net mixing parameter. 
- `batch_size`: The number of samples in each batch.
- `fit_intercept=true`: If True, the model tries to correct for the global mean of y. 
                        If False, the model expects that you have centered the data.
- `epochs=1000`: The number of times the model should iterate through the entire dataset during training.
- `tol=1e-3': The training process will stop if current_loss > previous_loss - tol.
- `shuffle=true`: If true, shuffles the training data after each epoch.
- `eta0=1e-3`: The initial learning rate.
- `power_t=0.5`: The exponent used for calculating the invscaling learning rate.
- `learning_rate="constant`: Method for modifying the learning rate during training
                            ("adaptive", "constant", "invscaling", "optimal")
    - `optimal`: not supported
    - `constant`: constant learning rate
    - `adaptive`: changes the learning rate if the training loss or the validation accuracy does 
                   not improve for n_iter_no_change epochs. The old learning rate is generally divided by 5.
    - `invscaling`: `eta = eta0 / pow(t, power_t)`
- `n_iter_no_change=5`: the number of epochs to train without any imporvement in the model
- `verbose=false`: Sets logging level.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are class assignments. 

- `predict_proba(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are probabilistic, but uncalibrated.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:

- `classes_seen`: list of target classes actually observed in training

- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)

# Examples
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
MBSGDClassifier


"""
$(MMI.doc_header(KNeighborsClassifier))

`KNeighborsClassifier`  is a wrapper for the RAPIDS K-Nearest Neighbors Classifier.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

- `y`: is the target, which can be any `AbstractVector` whose element
    scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
    with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `n_neighbors=5`: Default number of neighbors to query.
- `algorithm="brute"`: Only one algorithm is currently supported.
- `metric="euclidean"`: Distance metric to use.
- `weights="uniform"`: Sample weights to use. Currently, only the uniform strategy is supported.
- `verbose=false`: Sets logging level.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are class assignments. 

- `predict_proba(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are probabilistic, but uncalibrated.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:

- `classes_seen`: list of target classes actually observed in training

- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)

# Examples
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
KNeighborsClassifier