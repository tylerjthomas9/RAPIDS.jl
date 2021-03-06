"""
    module RAPIDS

A Julia interface to the RAPIDS AI ecosystem
"""
module RAPIDS

using CUDA
using MLJBase
using MLJModelInterface

const MMI = MLJModelInterface
const PKG = "RAPIDS"

if !CUDA.functional()
    @warn "No CUDA GPU Detected. Unable to load RAPIDS."
    const VERSION = VersionNumber(0,1,0) # fake version number for automerge
    export
    VERSION
    
else
    using PythonCall
    const cudf = PythonCall.pynew()
    #const cuxfilter = PythonCall.pynew() #TODO fix error during import
    const cugraph = PythonCall.pynew()
    const cuml = PythonCall.pynew()
    const cupy = PythonCall.pynew()
    const cusignal = PythonCall.pynew()
    const cuspatial = PythonCall.pynew()
    const dask = PythonCall.pynew()
    const dask_cuda = PythonCall.pynew()
    const dask_cudf = PythonCall.pynew()
    const numpy = PythonCall.pynew()
    const pickle = PythonCall.pynew()

    function __init__()
        PythonCall.pycopy!(cudf, pyimport("cudf"))
        #PythonCall.pycopy!(cuxfilter, pyimport("cuxfilter"))
        PythonCall.pycopy!(cugraph, pyimport("cugraph"))
        PythonCall.pycopy!(cuml, pyimport("cuml"))
        PythonCall.pycopy!(cusignal, pyimport("cusignal"))
        PythonCall.pycopy!(cupy, pyimport("cupy"))
        PythonCall.pycopy!(cuspatial, pyimport("cuspatial"))
        PythonCall.pycopy!(dask, pyimport("dask"))
        PythonCall.pycopy!(dask_cuda, pyimport("dask_cuda"))
        PythonCall.pycopy!(dask_cudf, pyimport("dask_cudf"))
        PythonCall.pycopy!(numpy, pyimport("numpy"))
        PythonCall.pycopy!(pickle, pyimport("pickle"))
    end


    include("./mlj_interface.jl")


    export
    # RAPIDS Python API
    cudf, 
    #cuxfilter,
    cugraph,
    cuml,
    cusignal,
    cupy,
    cuspatial,
    dask,
    dask_cuda,
    dask_cudf,
    numpy,

    # PythonCall
    pycopy!,
    pyimport,
    pynew,

    # clustering
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    HDBSCAN,
    # regression
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    MBSGDRegressor,
    RandomForestRegressor,
    CD,
    SVR,
    KNeighborsRegressor,
    # classification
    LogisticRegression,
    MBSGDClassifier,
    KNeighborsClassifier,
    # dimensionality reduction
    PCA,
    IncrementalPCA,
    TruncatedSVD,
    UMAP,
    GaussianRandomProjection,
    TSNE,
    # time series
    ExponentialSmoothing, 
    forecast
end


end