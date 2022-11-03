using CUDA
using RAPIDS
using MLJBase
using Test

if CUDA.functional()
    include("./cuml.jl")
end
