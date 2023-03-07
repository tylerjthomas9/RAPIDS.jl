# MLJ API

# MLJ Example - Classification

```julia
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)
y = [repeat([0], 50)..., repeat([1], 50)...]

model = LogisticRegression()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)

println(mach.fitresult.coef_)
```

# MLJ Example - Regression

```julia
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)
y = rand(100)

model = LinearRegression()
mach = machine(model, X, y)
fit!(mach)
preds = predict(mach, X)

println(mach.fitresult.coef_)
```


## Clustering
```@docs
CuML.KMeans
CuML.DBSCAN
CuML.AgglomerativeClustering
CuML.HDBSCAN
```

## Classification
```@docs
CuML.LogisticRegression
CuML.MBSGDClassifier
CuML.RandomForestClassifier
CuML.SVC
CuML.LinearSVC
CuML.KNeighborsClassifier
```

## Regression
```@docs
CuML.LinearRegression
CuML.Ridge
CuML.Lasso
CuML.ElasticNet
CuML.MBSGDRegressor
CuML.RandomForestRegressor
CuML.CD
CuML.SVR
CuML.LinearSVR
CuML.KNeighborsRegressor
```

## Dimensionality Reduction
```@docs
CuML.PCA
CuML.IncrementalPCA
CuML.TruncatedSVD
CuML.UMAP
CuML.TSNE
CuML.GaussianRandomProjection
```

## Time Series
```@docs
CuML.ExponentialSmoothing
CuML.ARIMA
```