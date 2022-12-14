
module CUML

using MLJBase
using MLJModelInterface
using RAPIDS: numpy, pickle, cuml
using Reexport
using PythonCall
using Tables

const MMI = MLJModelInterface

include("utils.jl")

include("classification.jl")
include("clustering.jl")
include("dimensionality_reduction.jl")
include("regression.jl")
include("time_series.jl")


const CUML_MODELS = Union{
    CUML_CLASSIFICATION,
    CUML_CLUSTERING,
    CUML_DIMENSIONALITY_REDUCTION,
    CUML_REGRESSION,
    CUML_TIME_SERIES,
}

MMI.clean!(model::CUML_MODELS) = ""

# MLJ Package Metadata
MMI.package_name(::Type{<:CUML_MODELS}) = "RAPIDS"
MMI.package_uuid(::Type{<:CUML_MODELS}) = "2764e59e-7dd7-4b2d-a28d-ce06411bac13"
MMI.package_url(::Type{<:CUML_MODELS}) = "https://github.com/tylerjthomas9/RAPIDS.jl"
MMI.is_pure_julia(::Type{<:CUML_MODELS}) = false

# Feature Importances
MMI.reports_feature_importances(::Type{<:CUML_MODELS}) = false #TODO: add feature importance
MMI.supports_weights(::Type{<:CUML_MODELS}) = false #TODO: add weights support

include("mlj_serialization.jl")

# docstrings
include("./cuml/classification_docstrings.jl")
include("./cuml/clustering_docstrings.jl")
include("./cuml/dimensionality_reduction_docstrings.jl")
include("./cuml/regression_docstrings.jl")
include("./cuml/time_series_docstrings.jl")

export 
# helper functions
to_numpy,

# Types
CUML_CLASSIFICATION,
CUML_CLUSTERING,
CUML_DIMENSIONALITY_REDUCTION,
CUML_REGRESSION,
CUML_TIME_SERIES,

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
LinearSVR,
KNeighborsRegressor,
# classification
LogisticRegression,
MBSGDClassifier,
RandomForestClassifier,
SVC,
LinearSVC,
KNeighborsClassifier,
# dimensionality reduction
PCA,
IncrementalPCA,
TruncatedSVD,
UMAP,
GaussianRandomProjection,
SparseRandomProjection,
TSNE,
# time series
ExponentialSmoothing,
ARIMA,
forecast

end