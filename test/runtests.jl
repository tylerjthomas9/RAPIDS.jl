using CUDA

if CUDA.functional()
    using Aqua
    using MLJBase
    using MLJTestInterface
    using RAPIDS
    using RAPIDS.CuML
    using RAPIDS.CuDF
    using Tables
    using Test

    # include("cudf.jl")
    include("cuml.jl")
    include("cuml_integration.jl")

    Aqua.test_all(RAPIDS; ambiguities=false)
else
    @warn "Skipping tests because a CUDA compatible GPU was not detected."
end
