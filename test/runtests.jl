using RAPIDS
using MLJ
using CUDA
using Test

if CUDA.functional()
    include("./cuml.jl")
end
