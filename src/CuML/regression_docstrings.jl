
"""
$(MMI.doc_header(LinearRegression))

`LinearRegression` is a wrapper for the RAPIDS Linear Regression.

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
- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS.CuML
using MLJBase

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
$(MMI.doc_header(Ridge))

`Ridge` is a wrapper for the RAPIDS Ridge Regression.

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
- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)
y = rand(100)

model = Ridge()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
Ridge

"""
$(MMI.doc_header(Lasso))

`Lasso` is a wrapper for the RAPIDS Lasso Regression.

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
- `tol=1e-4`: Tolerance for stopping criteria. 
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
- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)
y = rand(100)

model = Lasso()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
Lasso

"""
$(MMI.doc_header(ElasticNet))

`ElasticNet` is a wrapper for the RAPIDS ElasticNet Regression.

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
- `l1_ratio=0.5`: The ElasticNet mixing parameter.
- `fit_intercept=true`: If true, the model tries to correct for the global mean of y. 
                        If false, the model expects that you have centered the data.
- `normalize=true`: This parameter is ignored when fit_intercept is set to false.
                    If true, the predictors in X will be normalized by dividing by the column-wise standard deviation. 
- `max_iter=1000`: Maximum number of iterations taken for the solvers to converge.
- `tol=1e-3`: Tolerance for stopping criteria. 
- `solver="cd"`: 
    - `cd`: Coordinate descent.
    - `qn`: quasi-newton. You may find the alternative ‘qn’ algorithm is faster when the number of features is sufficiently large, but the sample size is small.
- `fit_intercept=true`: If true, the model tries to correct for the global mean of y. 
                        If false, the model expects that you have centered the data.
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
- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)
y = rand(100)

model = ElasticNet()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
ElasticNet

"""
$(MMI.doc_header(MBSGDRegressor))

`MBSGDRegressor` is a wrapper for the RAPIDS MBSGDRegressor.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`
- `y`: is an `AbstractVector` continuous target.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `loss="squared_loss`: `squared_loss` uses linear regression
- `penalty="none`: 
    - `none`: No regularization.
    - `l1`: L1 norm (Lasso).
    - `l2`: L2 norm (Ridge).
    - `elasticnet`: weighted average of L1 and L2 norms.
- `alpha=0.0001`: The constant value which decides the degree of regularization.
- `fit_intercept=true`: If true, the model tries to correct for the global mean of y. 
                        If false, the model expects that you have centered the data.
- `l1_ratio=0.5`: The ElasticNet mixing parameter.
- `batch_size=32`: The number of samples in each batch.
- `epochs=1000`: The number of times the model should iterate through the entire dataset.
- `tol=1e-3`: Tolerance for stopping criteria.
- `shuffle=true`: True, shuffles the training data after each epoch False, does not shuffle the training data after each epoch
- `eta_0=0.001`: Initial learning rate.
- `power_t=0.5`: The exponent used for calculating the invscaling learning rate.
- `learning_rate="constant`:
- `n_iter_no_change=5`: The number of epochs to train without any improvement in the model.
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
- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)
y = rand(100)

model = MBSGDRegressor()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
MBSGDRegressor

"""
$(MMI.doc_header(RandomForestRegressor))

`RandomForestRegressor` is a wrapper for the RAPIDS RandomForestRegressor.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`
- `y`: is an `AbstractVector` continuous target.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `n_estimators=100`: The total number of trees in the forest.
- `split_creation=2`: The criterion used to split nodes
    - `2` or `mse` for mean squared error
    - `4` or `poisson` for poisson half deviance
    - `5` or `gamma` for gamma half deviance
    - `6` or `inverse_gaussian` for inverse gaussian deviance
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
- `accuracy_metric="r2"`
    - `r2`: r-squared
    - `median_ae`: median of absolute error
    - `mean_ae`: mean of absolute error
    - `mse`: mean squared error
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
- `features`: the names of the features encountered in training., in an
  order consistent with the output of `print_tree` (see below)

# Examples
```
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)
y = rand(100)

model = RandomForestRegressor()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
RandomForestRegressor

"""
$(MMI.doc_header(CD))

`CD` is a wrapper for the RAPIDS Coordinate Descent.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`
- `y`: is an `AbstractVector` continuous target.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `alpha=0.0001`: Regularization strength. alpha = 0 is equivalent to an ordinary least square.
- `l1_ratio=0.15`: The ElasticNet mixing parameter.
- `fit_intercept=true`: If true, the model tries to correct for the global mean of y. 
                        If false, the model expects that you have centered the data.
- `normalize=false`: If true, the data is normalized.
- `tol=0.001`: The tolerance for the optimization: if the updates are smaller than tol, solver stops.
- `shuffle=true`: If true, a random coefficient is updated at each iteration. 
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
- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)
y = rand(100)

model = CD()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
CD

"""
$(MMI.doc_header(SVR))

`SVR` is a wrapper for the RAPIDS SVM Regressor.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`
- `y`: is an `AbstractVector` continuous target.

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
- `max_iter=-1`: Limit the number of outer iterations in the solver. If `-1` (default) then `max_iter=100*n_samples`.
- `nochange_steps=1000`: Stop training if a `1e-3*tol` difference isn't seen in `nochange_steps` steps.
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
- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)
y = rand(100)

model = SVR()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
SVR

"""
$(MMI.doc_header(LinearSVR))

`SVR` is a wrapper for the RAPIDS Linear SVM Regressor.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`
- `y`: is an `AbstractVector` continuous target.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `penalty="l2`: `l1` (Lasso) or `l2` (Ridge) penalty.
- `loss="epsilon_insensitive"`: The loss term of the target function.
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
- `epsilon=0.0`: The epsilon-sensitivity parameter for the SVR loss function.
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
- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)
y = rand(100)

model = LinearSVR()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
LinearSVR

"""
$(MMI.doc_header(KNeighborsRegressor))

`KNeighborsRegressor` is a wrapper for the RAPIDS K-Nearest Neighbors Regressor.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`
- `y`: is an `AbstractVector` continuous target.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `n_neighbors=5`: Default number of neighbors to query.
- `algorithm="auto"`: The query algorithm to use. 
- `metric="euclidean"`: Distance metric to use.
- `weights="uniform"`: Sample weights to use. Currently, only the uniform strategy is supported.
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
- `features`: the names of the features encountered in training.

# Examples
```
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)
y = rand(100)

model = KNeighborsRegressor()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```
"""
KNeighborsRegressor
