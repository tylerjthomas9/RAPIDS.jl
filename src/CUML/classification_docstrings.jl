
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
    features `Xnew` having the same scitype as `X` above.  Predictions
    are probabilistic.

- `predict_mean(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are classes.

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
    features `Xnew` having the same scitype as `X` above.  Predictions
    are probabilistic.

- `predict_mean(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are classes.

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
    features `Xnew` having the same scitype as `X` above.  Predictions
    are probabilistic.

- `predict_mean(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are classes.

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
    features `Xnew` having the same scitype as `X` above.  Predictions
    are probabilistic.

- `predict_mean(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are classes.

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
    features `Xnew` having the same scitype as `X` above.  Predictions
    are probabilistic.

- `predict_mean(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are classes.

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
