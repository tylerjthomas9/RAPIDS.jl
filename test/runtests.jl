using CUDA
using RAPIDS
using RAPIDS.CuML
using MLJBase
using MLJTestIntegration
using Test

if CUDA.functional()
    include("cudf.jl")
    include("cuml.jl")
    # include("cuml_integration.jl")
else
    @warn "Skipping tests for CI, Docs."
end
