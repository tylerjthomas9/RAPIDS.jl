
"""
    to_numpy(x; dtype::DataType=Float32)

Convert a array or table in julia to a numpy array

Parameters
----------
- x: Array or table

Returns
-------
- numpy array
"""
function to_numpy(x; dtype::DataType=Float32)
    x = dtype.(MMI.matrix(x))
    return numpy.array(x)
end

function to_numpy(x::AbstractVector{<:Real}; dtype=Float32)
    return numpy.array(dtype.(x))
end

function mlj_to_kwargs(model)
    return Dict{Symbol,Any}(
        name => getfield(model, name) for name in fieldnames(typeof(model))
    )
end

function _feature_names(X)
    schema = Tables.schema(X)
    if schema === nothing
        features = [Symbol("x$j") for j = 1:size(X, 2)]
    else
        features = collect(schema.names)
    end
    return features
end

include("./cuml/classification.jl")
include("./cuml/clustering.jl")
include("./cuml/dimensionality_reduction.jl")
include("./cuml/regression.jl")
include("./cuml/time_series.jl")

const CUML_MODELS = Union{
    CUML_CLASSIFICATION,
    CUML_CLUSTERING,
    CUML_DIMENSIONALITY_REDUCTION,
    CUML_REGRESSION,
    CUML_TIME_SERIES,
}
include("./cuml/mlj_serialization.jl")

MMI.clean!(model::CUML_MODELS) = ""

# MLJ Package Metadata
MMI.package_name(::Type{<:CUML_MODELS}) = "RAPIDS"
MMI.package_uuid(::Type{<:CUML_MODELS}) = "2764e59e-7dd7-4b2d-a28d-ce06411bac13"
MMI.package_url(::Type{<:CUML_MODELS}) = "https://github.com/tylerjthomas9/RAPIDS.jl"
MMI.is_pure_julia(::Type{<:CUML_MODELS}) = false
