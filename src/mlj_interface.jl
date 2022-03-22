include("./cuml/kmeans.jl")


const CUML_UNSUPERVISED = Union{cuKMeans, }

function mlj_to_kwargs(model::CUML_UNSUPERVISED)
    return Dict{Symbol, Any}(
        name => getfield(model, name)
        for name in fieldnames(typeof(model))
    )
end


function MMI.fit(mlj_model::CUML_UNSUPERVISED, verbosity, X, w=nothing)
    # initialize model, prepare data
    model = model_init(mlj_model)
    X = MMI.matrix(X) .|> Float32

    # fit the model
    model.fit(X)
    fitresult = (model, deepcopy(mlj_model))

    # save result
    cache = nothing
    n_iter = model.n_iter_
    report = (n_iter=pyconvert(Int64, model.n_iter_), 
            labels=pyconvert(Vector{Int64}, model.labels_),
            cluster_centers = pyconvert(Matrix{Float32}, model.cluster_centers_)
    )
    return (fitresult, cache, report)
end

function MMI.predict(mlj_model::CUML_UNSUPERVISED, fitresult, Xnew)
    model, _ = fitresult
    Xnew = MMI.matrix(Xnew) .|> Float32
    py_preds = model.predict(Xnew)
    preds = pyconvert(Vector{Int}, py_preds) 

    return preds
end
    

MMI.metadata_pkg.(CUML_UNSUPERVISED,
    name = "cuml",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url  = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,      # does it wrap around some other package?
)