"""
    module RAPIDS

A Julia interface to the RAPIDS AI ecosystem
"""
module RAPIDS

using PythonCall
using MLJModelInterface

const cudf = PythonCall.pynew()
const cuml = PythonCall.pynew()
const cugraph = PythonCall.pynew()
const cusignal = PythonCall.pynew()
const cuspatial = PythonCall.pynew()
const cuxfilter = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(cudf, pyimport("cudf"))
    PythonCall.pycopy!(cuml, pyimport("cuml"))
    PythonCall.pycopy!(cugraph, pyimport("cugraph"))
    PythonCall.pycopy!(cusignal, pyimport("cusignal"))
    PythonCall.pycopy!(cuspatial, pyimport("cuspatial"))
    PythonCall.pycopy!(cuxfilter, pyimport("cuxfilter"))
end


include("./cuml.jl")

const ALL_MODELS = Union{cuKMeans, }

MLJModelInterface.metadata_pkg.(ALL_MODELS,
    name = "RAPIDS",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url  = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",       # your package license
    is_wrapper = true,    # does it wrap around some other package?
)

end