using CUDA

if CUDA.functional()
    using MLJBase
    using MLJTestIntegration
    using RAPIDS
    using RAPIDS.CuML
    using Tables
    using Test

    include("cudf.jl")
    include("cuml.jl")
    include("cuml_integration.jl")
else
    @warn "Skipping tests because a CUDA compatible GPU was not detected."
end
