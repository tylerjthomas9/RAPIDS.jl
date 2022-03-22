"""
    module RAPIDS

A Julia interface to the RAPIDS AI ecosystem
"""
module RAPIDS

using PythonCall
using MLJBase
using MLJModelInterface
const MMI = MLJModelInterface


const cudf = PythonCall.pynew()
const cuxfilter = PythonCall.pynew()
const cugraph = PythonCall.pynew()
const cuml = PythonCall.pynew()
const cupy = PythonCall.pynew()
const cusignal = PythonCall.pynew()
const cuspatial = PythonCall.pynew()
const dask = PythonCall.pynew()
const dask_cuda = PythonCall.pynew()
const dask_cudf = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(cudf, pyimport("cudf"))
    PythonCall.pycopy!(cuxfilter, pyimport("cuxfilter"))
    PythonCall.pycopy!(cugraph, pyimport("cugraph"))
    PythonCall.pycopy!(cuml, pyimport("cuml"))
    PythonCall.pycopy!(cusignal, pyimport("cusignal"))
    PythonCall.pycopy!(cupy, pyimport("cupy"))
    PythonCall.pycopy!(cuspatial, pyimport("cuspatial"))
    PythonCall.pycopy!(dask, pyimport("dask"))
    PythonCall.pycopy!(dask_cuda, pyimport("dask_cuda"))
    PythonCall.pycopy!(dask_cudf, pyimport("dask_cudf"))
end


include("./mlj_interface.jl")


export
# RAPIDS Python API
cudf, 
cuxfilter,
cugraph,
cuml,
cusignal,
cupy,
cuspatial,
dask,
dask_cuda,
dask_cudf,

# PythonCall
pycopy!,
pyimport,
pynew,

# MLJ Interface
cuKMeans

end