using CUDA
using RAPIDS
using MLJBase
using MLJTestIntegration
using Test

if CUDA.functional()
    include("./cuml.jl")
    include("./cuml_integration.jl")
else
    @warn "Skipping tests for CI, Docs."
end
