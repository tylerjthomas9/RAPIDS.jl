[![CI](https://github.com/tylerjthomas9/RAPIDS.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/tylerjthomas9/RAPIDS.jl/actions/workflows/ci.yml)

# RAPIDS.jl
Julia wrapper for the [RAPIDS AI](https://rapids.ai/index.html) ecosystem. Support is limited to Linux.

The goal of this library is to provide a simple method for accessing the GPU accelerated models withing RAPIDS from Julia, and integrating the models into MLJ. This library relies on [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl) and [CondaPkg.jl](https://github.com/cjdoris/CondaPkg.jl) for efficient installations of the Python dependencies. 