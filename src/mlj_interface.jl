function prepare_input(x::AbstractMatrix)
    x = MMI.matrix(x) .|> Float32
    return numpy.array(x)
end

function prepare_input(x::AbstractVector)
    x = x .|> Float32 |> numpy.array
    return numpy.array(x)
end

function mlj_to_kwargs(model)
    return Dict{Symbol, Any}(
        name => getfield(model, name)
        for name in fieldnames(typeof(model))
    )
end



include("./cuml/classification.jl")
include("./cuml/clustering.jl")
include("./cuml/regression.jl")

const CUML_MODELS = Union{CUML_CLASSIFICATION, 
                            CUML_CLUSTERING, 
                            CUML_REGRESSION}

include("./cuml/serialization.jl")