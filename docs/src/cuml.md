# MLJ Example - Classification

```julia
using RAPIDS
using MLJBase

X = rand(100, 5)
y = [repeat([0], 50)..., repeat([1], 50)...]

model = LogisticRegression()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```

# MLJ Example - Regression

```julia
using RAPIDS
using MLJBase

X = rand(100, 5)
y = rand(100)

model = LinearRegression()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)
```


# MLJ API
```@meta
CurrentModule = RAPIDS
```

```@index
```
## Clustering
```@docs
    - `KMeans`
    - `DBSCAN`
    - `AgglomerativeClustering`
    - `HDBSCAN`
```

## Classification
```@docs
LogisticRegression
MBSGDClassifier
RandomForestClassifier
SVC
LinearSVC
KNeighborsClassifier
```

## Regression
```@docs
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
```

## Dimensionality Reduction
```@docs
    - `PCA`
    - `IncrementalPCA`
    - `TruncatedSVD`
    - `UMAP`
    - `TSNE`
    - `GaussianRandomProjection`
```

## Time Series
```@docs
    - `ExponentialSmoothing`
    - `ARIMA`
```