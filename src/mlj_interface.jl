include("./cuml.jl")

const CUML_MODELS = Union{cuKMeans, }

function mlj_to_kwargs(model::CUML_MODELS)
    return Dict{Symbol, Any}(
        name => getfield(model, name)
        for name in fieldnames(typeof(model))
    )
end

function MMI.fit(mlj_model::cuKMeans, verbosity, X, w=nothing)
    model = model_init(mlj_model)
    X = MMI.matrix(X)

    # cuml wants Float32
    X = Float32.(X)

    model.fit(X)
    fitresult = (model, deepcopy(mlj_model))

    cache = nothing
    n_iter = m.n_iter_
    report = (n_iter=pyconvert(Int64, model.n_iter_), 
            labels=pyconvert(Vector{Int64}, model.labels_),
            cluster_centers = pyconvert(Matrix{Float32}, model.cluster_centers_)
    )
    return (fitresult, cache, report)
end

# Then for each model,
MMI.metadata_model(cuKMeans,
    input_scitype   = Matrix,  # what input data is supported?
    output_scitype  = Matrix,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's KMeans: https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering",
	load_path    = "RAPIDS.cuml.KMeans"
)


MMI.metadata_pkg.(CUML_MODELS,
    name = "cuml",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url  = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,      # does it wrap around some other package?
)

