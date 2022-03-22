[![CI](https://github.com/tylerjthomas9/RAPIDS.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/tylerjthomas9/RAPIDS.jl/actions/workflows/ci.yml)

# RAPIDS.jl
Julia wrapper for the [RAPIDS AI](https://rapids.ai/index.html) ecosystem. Support is limited to Linux.

The goal of this library is to provide a simple method for accessing the GPU accelerated models withing RAPIDS from Julia, and integrating the models into MLJ. This library relies on [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl) and [CondaPkg.jl](https://github.com/cjdoris/CondaPkg.jl) for efficient installations of the Python dependencies. 

This wrapper could be broken up insto several libraries (`cuDF`, `cuML`, `cuGraph`, `cuSignal`, `cuSpatial`, `cuxfilter`), but there would be significant overlap between these libraries. Large dependencies such as `cudatoolkit` would be repeated.

## Python API

You can access the following python libraries with their standard syntax:
- `cupy`
- `cudf`
- `cuml`
- `cugraph`
- `cusignal`
- `cuspatial`
- `cuxfilter`
- `dask`
- `dask_cuda`
- `dask_cudf`
- `numpy`

Here is an example of using `KMeans`, `cudf` via the Python API. This example is taken from the [cuml GitHub](https://github.com/rapidsai/cuml)

```julia
using RAPIDS

# Create and populate a GPU DataFrame
gdf_float = cudf.DataFrame()
gdf_float['0'] = [1.0, 2.0, 5.0]
gdf_float['1'] = [4.0, 2.0, 1.0]
gdf_float['2'] = [4.0, 2.0, 1.0]

# Setup and fit clusters
dbscan = cuml.DBSCAN(eps=1.0, min_samples=1)
dbscan.fit(gdf_float)

print(dbscan.labels_)
```

## MLJ Interface

A MLJ interface is also available for supported models. The model hyperparameters are the same as described in the [cuML docs](https://docs.rapids.ai/api/cuml/stable/api.html). The only difference is that the models will always input/output numpy arrays, which will be converted back to Julia arrays (`output_type="input"`). 

```
using RAPIDS
using MLJ

x = rand(100, 5)

kmeans = cuKMeans()
mach = machine(kmeans, x)
fit!(mach)
preds = predict(mach, x)
```

MLJ Support:
- Clustering
    - `cuKMeans`
    - `cuDBSCAN`
    - `cuAgglomerativeClustering`
    - `cuHDBSCAN`
- Classification
- Regression