
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
    learning_rate::String = "constant"::(_ in
                                         ("adaptive", "constant", "invscaling", "optimal"))
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

const CUML_CLASSIFICATION = Union{LogisticRegression,MBSGDClassifier,
                                  RandomForestClassifier,SVC,
                                  LinearSVC,KNeighborsClassifier}

# add metadata
MMI.load_path(::Type{<:LogisticRegression}) = "$PKG.LogisticRegression"
MMI.load_path(::Type{<:MBSGDClassifier}) = "$PKG.MBSGDClassifier"
MMI.load_path(::Type{<:RandomForestClassifier}) = "$PKG.RandomForestClassifier"
MMI.load_path(::Type{<:SVC}) = "$PKG.SVC"
MMI.load_path(::Type{<:LinearSVC}) = "$PKG.LinearSVC"
MMI.load_path(::Type{<:KNeighborsClassifier}) = "$PKG.KNeighborsClassifier"

function MMI.input_scitype(::Type{<:CUML_CLASSIFICATION})
    return Union{Table(Continuous),AbstractMatrix{<:Continuous}}
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
function MMI.fit(mlj_model::CUML_CLASSIFICATION, verbosity, X, y, w=nothing)
    X_numpy = prepare_input(X)
    y_numpy = prepare_input(y)

    # fit the model
    model = model_init(mlj_model)
    model.fit(X_numpy, y_numpy)
    fitresult = model

    # save result
    cache = nothing
    y_cat = MMI.categorical(y)
    classes_seen = filter(in(unique(y_cat)), MMI.classes(y_cat))
    report = (classes_seen=classes_seen,
              features=_feature_names(X))
    return (fitresult, cache, report)
end

# predict methods
function MMI.predict(mlj_model::CUML_CLASSIFICATION, fitresult, Xnew)
    model = fitresult
    py_preds = model.predict(prepare_input(Xnew))
    preds = MMI.categorical(pyconvert(Array, py_preds))

    return preds
end

# Classification metadata
MMI.metadata_pkg.((LogisticRegression, MBSGDClassifier, RandomForestClassifier,
                   SVC, LinearSVC, KNeighborsClassifier),
                  name="cuML Classification Methods",
                  uuid="2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
                  url="https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
                  julia=false,          # is it written entirely in Julia?
                  license="MIT",        # your package license
                  is_wrapper=true)

# docstrings
# TODO: add Table/DataFrame examples

"""
$(MMI.doc_header(LogisticRegression))

`LogisticRegression` is a wrapper for the RAPIDS Logistic Regression.

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

- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS
using MLJBase

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

`MBSGDClassifier` is a wrapper for the RAPIDS Mini Batch SGD Classifier.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

- `y`: is an `AbstractVector` finite target.

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

- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS
using MLJBase

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
$(MMI.doc_header(RandomForestClassifier))

`RandomForestClassifier` is a wrapper for the RAPIDS RandomForestClassifier.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`
- `y`: is an `AbstractVector` finite target.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `n_estimators=100`: The total number of trees in the forest.
- `split_creation=2`: The criterion used to split nodes
    - `0` or `gini` for gini impurity
    - `1` or `entropy` for information gain (entropy)
- `bootstrap=true`: If true, each tree in the forest is built using a bootstrap sample with replacement.
- `max_samples=1.0`: Ratio of dataset rows used while fitting each tree.
- `max_depth=16`: Maximum tree depth.
- `max_leaves=-1`: Maximum leaf nodes per tree. Soft constraint. Unlimited, If `-1`.
- `max_features="auto"`: Ratio of number of features (columns) to consider per node split.
    - If type `Int` then max_features is the absolute count of features to be used.
    - If type `Float64` then `max_features` is a fraction.
    - If `auto` then `max_features=n_features = 1.0`.
    - If `sqrt` then `max_features=1/sqrt(n_features)`.
    - If `log2` then `max_features=log2(n_features)/n_features`.
    - If None, then `max_features=1.0`.
- `n_bins=128`: Maximum number of bins used by the split algorithm per feature.
- `n_streams=4`: Number of parallel streams used for forest building
- `min_samples_leaf=1`: The minimum number of samples in each leaf node.
    - If type `Int`, then `min_samples_leaf` represents the minimum number.
    - If `Float64`, then `min_samples_leaf` represents a fraction and `ceil(min_samples_leaf * n_rows) `
        is the minimum number of samples for each leaf node.
- `min_samples_split=2`: The minimum number of samples required to split an internal node.
    - If type `Int`, then `min_samples_split` represents the minimum number.
    - If `Float64`, then `min_samples_split` represents a fraction and `ceil(min_samples_leaf * n_rows) `
        is the minimum number of samples for each leaf node.
- `min_impurity_decrease=0.0`: The minimum decrease in impurity required for node to be split.
- `max_batch_size=4096`: Maximum number of nodes that can be processed in a given batch.
- `random_state=nothing`: Seed for the random number generator.
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

- `classes_seen`: list of target classes actually observed in training

- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS
using MLJBase

X = rand(100, 5)
y = [repeat([0], 50)..., repeat([1], 50)...]

model = RandomForestClassifier()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
RandomForestClassifier

"""
$(MMI.doc_header(SVC))

`SVC` is a wrapper for the RAPIDS SVC.

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

- `C=1.0`: The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
- `kernel="rbf"`: `linear`, `poly`, `rbf`, `sigmoid` are supported.
- `degree=3`: Degree of polynomial kernel function.
- `gamma="scale"`
    - `auto`: gamma will be set to `1 / n_features`
    - `scale`: gamma will be set to `1 / (n_features * var(X))`
- `coef0=0.0`: Independent term in kernel function, only signifficant for poly and sigmoid.
- `tol=0.001`: Tolerance for stopping criterion.
- `cache_size=1024.0`: Size of the cache during training in MiB.
- `class_weight=nothing`: Weights to modify the parameter C for class i to `class_weight[i]*C`. The string `"balanced"`` is also accepted.
- `max_iter=-1`: Limit the number of outer iterations in the solver. If `-1` (default) then `max_iter=100*n_samples`.
- `multiclass_strategy="ovo"`
    - `ovo`: OneVsOneClassifier
    - `ovr`: OneVsRestClassifier
- `nochange_steps=1000`: Stop training if a `1e-3*tol` difference isn't seen in `nochange_steps` steps.
- `probability=false`: Enable or disable probability estimates.
- `random_state=nothing`: Seed for the random number generator.
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

- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS
using MLJBase

X = rand(100, 5)
y = [repeat([0], 50)..., repeat([1], 50)...]

model = SVC()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
SVC

"""
$(MMI.doc_header(LinearSVC))

`LinearSVC` is a wrapper for the RAPIDS LinearSVC.

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

- `penalty="l2`: `l1` (Lasso) or `l2` (Ridge) penalty.
- `loss="squared_hinge"`: The loss term of the target function.
- `fit_intercept=true`: If true, the model tries to correct for the global mean of y. 
                        If false, the model expects that you have centered the data.
- `penalized_intercept=true`: When true, the bias term is treated the same way as other features.
- `max_iter=1000`: Maximum number of iterations for the underlying solver.
- `linesearch_max_iter=1000`: Maximum number of linesearch (inner loop) iterations for the underlying (QN) solver.
- `lbfgs_memory=5`: Number of vectors approximating the hessian for the underlying QN solver (l-bfgs).
- `C=1.0`: The constant scaling factor of the loss term in the target formula `F(X, y) = penalty(X) + C * loss(X, y)`.
- `grad_tol=0.0001`: The threshold on the gradient for the underlying QN solver.
- `change_tol=0.00001`: The threshold on the function change for the underlying QN solver.
- `tol=nothing`: Tolerance for stopping criterion.
- `probabability=false`: Enable or disable probability estimates.
- `multi_class="ovo"`
    - `ovo`: OneVsOneClassifier
    - `ovr`: OneVsRestClassifier
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

- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS
using MLJBase

X = rand(100, 5)
y = [repeat([0], 50)..., repeat([1], 50)...]

model = LinearSVC()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
LinearSVC

"""
$(MMI.doc_header(KNeighborsClassifier))

`KNeighborsClassifier` is a wrapper for the RAPIDS K-Nearest Neighbors Classifier.

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

- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS
using MLJBase

X = rand(100, 5)
y = [repeat([0], 50)..., repeat([1], 50)...]

model = KNeighborsClassifier()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
KNeighborsClassifier
