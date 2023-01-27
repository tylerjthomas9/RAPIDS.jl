"""
    module RAPIDS

A Julia interface to the RAPIDS AI ecosystem
"""
module RAPIDS

using CUDA

const PKG = "RAPIDS"

# Temp warning for 
if Base.VERSION <= v"1.8.3"
    warning_msg = """
    RAPIDS.jl does not work out of the box with Julia versions before v1.9.
    You must manually upgrade to libraries from GCC 12.
    """
    @warn warning_msg
end


if !CUDA.functional()
    @warn "No CUDA GPU Detected. Unable to load RAPIDS."
    const cudf = nothing
    const cugraph = nothing
    const cuml = nothing
    const cusignal = nothing
    const cuspatial = nothing
    const cuxfilter = nothing
    const cupy = nothing
    const dask = nothing
    const dask_cuda = nothing
    const dask_cudf = nothing
    const numpy = nothing
    const pickle = nothing
    abstract type Py end
    macro py(x...) end
else
    @info "CUDA GPU Detected"
    using PythonCall
    const cudf = PythonCall.pynew()
    const cugraph = PythonCall.pynew()
    const cuml = PythonCall.pynew()
    const cusignal = PythonCall.pynew()
    const cuspatial = PythonCall.pynew()
    const cuxfilter = PythonCall.pynew()
    const cupy = PythonCall.pynew()
    const dask = PythonCall.pynew()
    const dask_cuda = PythonCall.pynew()
    const dask_cudf = PythonCall.pynew()
    const numpy = PythonCall.pynew()
    const pickle = PythonCall.pynew()
    function __init__()
        PythonCall.pycopy!(cudf, pyimport("cudf"))
        PythonCall.pycopy!(cugraph, pyimport("cugraph"))
        PythonCall.pycopy!(cuml, pyimport("cuml"))
        PythonCall.pycopy!(cusignal, pyimport("cusignal"))
        PythonCall.pycopy!(cuspatial, pyimport("cuspatial"))
        PythonCall.pycopy!(cuxfilter, pyimport("cuxfilter"))
        PythonCall.pycopy!(cupy, pyimport("cupy"))
        PythonCall.pycopy!(dask, pyimport("dask"))
        PythonCall.pycopy!(dask_cuda, pyimport("dask_cuda"))
        PythonCall.pycopy!(dask_cudf, pyimport("dask_cudf"))
        PythonCall.pycopy!(numpy, pyimport("numpy"))
        return PythonCall.pycopy!(pickle, pyimport("pickle"))
    end
end

export VERSION,
    # PythonCall
    pycopy!,
    pyimport,
    pynew,

    # Python API
    cudf,
    # cugraph,
    cuml,
    # cusignal,
    # cuspatial,
    # cuxfilter,
    cupy,
    dask,
    dask_cuda,
    dask_cudf,
    numpy

    include("CuDF/CuDF.jl")
    include("CuML/CuML.jl")

end
