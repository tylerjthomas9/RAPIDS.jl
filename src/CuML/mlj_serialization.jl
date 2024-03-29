
# https://docs.rapids.ai/api/cuml/stable/pickling_cuml_models.html

"""
    persistent(booster)

Private method.

Return a persistent (ie, Julia-serializable) representation of the
RAPIDS.jl model.

Restore the model with [`model`](@ref)

"""
function persistent(model)
    model_file, io = mktemp()
    close(io)

    pickle.dump(model, @py open(model_file, "wb"))
    persistent_booster = read(model_file)
    rm(model_file)
    return persistent_booster
end

"""
    booster(persistent)

Private method.

Return the RAPIDS.jl model which has `persistent` as its persistent
(Julia-serializable) representation. See [`persistent`](@ref) method.
"""
function booster(persistent)
    model_file, io = mktemp()
    write(io, persistent)
    close(io)
    model = pickle.load(@py open(model_file, "rb"))
    rm(model_file)

    return model
end

MMI.save(::CUML_MODELS, fitresult; kwargs...) = persistent(fitresult)

MMI.restore(::CUML_MODELS, serializable_fitresult) = booster(serializable_fitresult)
