# TODO: Is there a way to generate docs without needing to
# generate them on a gpu node
using RAPIDS
using Documenter

Documenter.makedocs(; modules = [RAPIDS], sitename = "RAPIDS.jl")
