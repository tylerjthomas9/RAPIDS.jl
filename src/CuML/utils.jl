
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
