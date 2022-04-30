# RAPIDS.jl
Unofficial Julia wrapper for the [RAPIDS.ai](https://rapids.ai/index.html) ecosystem. Support is limited to Linux and NVIDIA GPUs (Pascal or more recent) with at least 6 TFLOPS.

The goal of this library is to provide a simple method for accessing the GPU accelerated models withing RAPIDS from Julia, and integrating the models into MLJ. This library relies on [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl) and [CondaPkg.jl](https://github.com/cjdoris/CondaPkg.jl) for efficient installations of the Python dependencies. 

This wrapper could be broken up into several libraries (`cuDF`, `cuML`, `cuGraph`, `cuSignal`, `cuSpatial`), but there would be significant overlap between these libraries. Large dependencies such as `cudatoolkit` would be repeated.

## Installation

From source:
```julia
julia> ]add https://github.com/tylerjthomas9/RAPIDS.jl
```

```julia
julia> using Pkg; Pkg.add(url="https://github.com/tylerjthomas9/RAPIDS.jl")
```

## Python API

You can access the following python libraries with their standard syntax:
- `cupy`
- `cudf`
- `cuml`
- `cugraph`
- `cusignal`
- `cuspatial`
- `dask`
- `dask_cuda`
- `dask_cudf`
- `numpy`
- `pickle`

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
using MLJ
using RAPIDS
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
