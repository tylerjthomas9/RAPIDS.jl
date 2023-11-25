[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.juliahub.com/RAPIDS/hxbio/0.2.0/)
[![Lifecycle:Maturing](https://img.shields.io/badge/Lifecycle-Maturing-007EC6)](https://github.com/bcgov/repomountie/blob/master/doc/lifecycle-badges.md)
[![Code Style: YASGuide](https://img.shields.io/badge/code%20style-yas-violet.svg)](https://github.com/jrevels/YASGuide)

# RAPIDS.jl
Unofficial Julia wrapper for the [RAPIDS.ai](https://rapids.ai/index.html) ecosystem.

The goal of this library is to provide a simple method for accessing the GPU accelerated models withing RAPIDS from Julia, and integrating the models into MLJ. This library relies on [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl) and [CondaPkg.jl](https://github.com/cjdoris/CondaPkg.jl) for efficient installations of the Python dependencies. 

This wrapper could be broken up into several libraries (`cuDF`, `cuML`, `cuGraph`, `cuSpatial`), but there would be significant overlap between these libraries. Large dependencies such as `cudatoolkit` would be repeated.

Long term, directly wrapping `libcudf`, `libcuml`... would greatly improve this library, but I don't have time to tackle that at this moment. 

# CUDA/GPU requirements
More information is available [here](https://docs.rapids.ai/install).
- CUDA 11.2+
- NVIDIA driver 470.42.01+
- Pascal architecture or better (Compute Capability >=6.0)
- Ubuntu 20.04/22.04 or CentOS 7 / Rocky Linux 8 with gcc/++ 9.0+

## Installation

From the Julia General Registry:
```julia
julia> ]  # enters the pkg interface
pkg> add RAPIDS
```

```julia
julia> using Pkg; Pkg.add("RAPIDS")
```

From source:
```julia
julia> ]add https://github.com/tylerjthomas9/RAPIDS.jl
```

```julia
julia> using Pkg; Pkg.add(url="https://github.com/tylerjthomas9/RAPIDS.jl")
```

## Julia Interfaces

- `CuDF`
- `CuML`

## Python API

You can access the following python libraries with their standard Python syntax:
- `cupy`
- `cudf`
- `cuml`
- `cugraph`
- `cuspatial`
- `cuxfilter`
- `dask`
- `dask_cuda`
- `dask_cudf`
- `numpy`
- `pandas` (cudf pandas)
- `pickle`


## Known Issues
- RAPIDS.jl is only supported on Julia 1.8.5+. For previous Julia versions, you have to manually upgrade to libraries from GCC 12. 
