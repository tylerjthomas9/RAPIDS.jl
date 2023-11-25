# CuML

## Python API

Here is an example of using `LogisticRegression`, `make_classification` via the Python API. 

```julia
using RAPIDS
const make_classification = cuml.datasets.classification.make_classification

X_py, y_py = make_classification(n_samples=200, n_features=4,
                           n_informative=2, n_classes=2)
lr = cuml.LogisticRegression(max_iter=100)
lr.fit(X_py, y_py)
preds = lr.predict(X_py)

print(lr.coef_)
```

## MLJ Interface

A MLJ interface is also available for supported models. The model hyperparameters are the same as described in the [cuML docs](https://docs.rapids.ai/api/cuml/stable/api.html). The only difference is that the models will always input/output numpy arrays, which will be converted back to Julia arrays (`output_type="input"`). 

```julia
using MLJBase
using RAPIDS.CuML
const make_classification = cuml.datasets.classification.make_classification

X_py, y_py = make_classification(n_samples=200, n_features=4,
                           n_informative=2, n_classes=2)
X = RAPIDS.pyconvert(Matrix{Float32}, X_py.get())
y = RAPIDS.pyconvert(Vector{Float32}, y_py.get().flatten())

lr = LogisticRegression(max_iter=100)
mach = machine(lr, X, y)
fit!(mach)
preds = predict(mach, X)

print(mach.fitresult.coef_)
```

MLJ Support:
- Clustering
    - `KMeans`
    - `DBSCAN`
    - `AgglomerativeClustering`
    - `HDBSCAN`
- Classification
    - `LogisticRegression`
    - `MBSGDClassifier`
    - `RandomForestClassifier`
    - `SVC`
    - `LinearSVC`
    - `KNeighborsClassifier`
- Regression
    - `LinearRegression`
    - `Ridge`
    - `Lasso`
    - `ElasticNet`
    - `MBSGDRegressor`
    - `RandomForestRegressor`
    - `CD`
    - `SVR`
    - `LinearSVR`
    - `KNeighborsRegressor`
- Dimensionality Reduction
    - `PCA`
    - `IncrementalPCA`
    - `TruncatedSVD`
    - `UMAP`
    - `TSNE`
    - `GaussianRandomProjection`
- Time Series
    - `ExponentialSmoothing`
    - `ARIMA`
