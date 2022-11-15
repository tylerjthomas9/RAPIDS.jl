# Python API

You can directly interface with the Python API for all RAPIDS packages. By default, the following packages are exported:
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
- `pickle`

# CUML Example - Classification
```julia
using RAPIDS
using PythonCall

X_numpy = numpy.random.rand(100, 5)
y_numpy = numpy.random.randint(0, 2, 100)

model = cuml.LogisticRegression()
model.fit(X_numpy, y_numpy)
preds_numpy = model.predict(X_numpy)
preds = pyconvert(Array, preds_numpy)
```

# CUML Example - Regression
```julia
using RAPIDS
using PythonCall

X_numpy = numpy.random.rand(100, 5)
y_numpy = numpy.random.rand(100)

model = cuml.LinearRegression()
model.fit(X_numpy, y_numpy)
preds_numpy = model.predict(X_numpy)
preds = pyconvert(Array, preds_numpy)
```