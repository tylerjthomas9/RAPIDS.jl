function prepare_x(x::AbstractMatrix)
    x = MMI.matrix(x) .|> Float32
    return numpy.array(x)
end

function prepare_y(y::AbstractVector)
    return numpy.array(Float32.(y))
end

function mlj_to_kwargs(model)
    return Dict{Symbol, Any}(
        name => getfield(model, name)
        for name in fieldnames(typeof(model))
    )
end


include("./cuml/clustering.jl")
include("./cuml/regression.jl")

