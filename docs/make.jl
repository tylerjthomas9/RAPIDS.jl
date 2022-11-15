using Documenter
using RAPIDS

DocMeta.setdocmeta!(RAPIDS, :DocTestSetup, :(using RAPIDS); recursive = true)

makedocs(;
    modules = [RAPIDS],
    authors = "tylerjthomas9 <tylerjthomas9@gmail.com>",
    repo = "https://github.com/tylerjthomas9/RAPIDS.jl.git",
    sitename = "RAPIDS.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://tylerjthomas9.github.io/RAPIDS.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md",
            "Python API" => "python_api.md",
            "cuMl" => "cuml.md",],
)

deploydocs(; repo = "github.com/tylerjthomas9/RAPIDS.jl.git")