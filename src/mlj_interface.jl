function prepare_input(x)
    x = Float32.(MMI.matrix(x))
    return numpy.array(x)
end

function RAPIDS.prepare_input(x::AbstractVector{<:Real})
    x = numpy.array(Float32.(x))
    return numpy.array(x)
end

function mlj_to_kwargs(model)
    return Dict{Symbol,Any}(name => getfield(model, name)
                            for name in fieldnames(typeof(model)))
end

function _feature_names(X)
    schema = Tables.schema(X)
    if schema === nothing
        features = [Symbol("x$j") for j in 1:size(X, 2)]
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

const CUML_MODELS = Union{CUML_CLASSIFICATION,
                          CUML_CLUSTERING,
                          CUML_DIMENSIONALITY_REDUCTION,
                          CUML_REGRESSION,
                          CUML_TIME_SERIES}
include("./cuml/mlj_serialization.jl")

MMI.clean!(model::CUML_MODELS) = ""

# MLJ Package Metadata
MMI.package_name(::Type{<:CUML_MODELS}) = "RAPIDS"
MMI.package_uuid(::Type{<:CUML_MODELS}) = "2764e59e-7dd7-4b2d-a28d-ce06411bac13"
MMI.package_url(::Type{<:CUML_MODELS}) = "https://github.com/tylerjthomas9/RAPIDS.jl"
MMI.is_pure_julia(::Type{<:CUML_MODELS}) = false
