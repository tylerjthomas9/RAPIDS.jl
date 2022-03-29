using RAPIDS
using MLJ
using Test

tests = ["cuml"]

println("Running tests:")
for t in tests
    fp = "$(t).jl"
    println("* $fp ...")
    include(fp)
end