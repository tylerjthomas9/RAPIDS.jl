

# Model hyperparameters

"""
RAPIDS AI Docs for KMeans: https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)

model = cuKMeans()
mach = machine(model, x)
fit!(mach)
preds = predict(mach, x)
```
"""
MLJModelInterface.@mlj_model mutable struct cuKMeans <: MMI.Unsupervised
    handle = nothing
    n_clusters::Int64 = 8::(_ > 0)
    max_iter::Int64 = 300::(_ > 0)
    tol::Float64 = 1e-4::(_ > 0)
    verbose::Bool = false
    random_state::Int = 1::(_ > 0)
    init::String = "scalable-k-means++"::(_ in ("scalable-k-means++","k-means||", "random"))
    n_init::Int = 1::(_ > 0)
    oversampling_factor::Float64 = 2.0::(_ > 0)
    max_samples_per_batch::Int64 = 32768::(_ > 0)
end

"""
RAPIDS AI Docs for DBSCAN: https://docs.rapids.ai/api/cuml/stable/api.html#dbscan


Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)

model = cuDBSCAN()
mach = machine(model, x)
fit!(mach)
preds = mach.report.labels #DBSCAN does not have a predict method
```
"""
MLJModelInterface.@mlj_model mutable struct cuDBSCAN <: MMI.Unsupervised
    handle = nothing
    eps::Float64 = 1e-4::(_ > 0)
    min_samples::Int = 1::(_ > 0)
    metric::String = "euclidean"::(_ in ("euclidean","precomputed"))
    verbose::Bool = false
    max_mbytes_per_batch = nothing
    calc_core_sample_indices::Bool = true
end

"""
RAPIDS AI Docs for AgglomerativeClustering: https://docs.rapids.ai/api/cuml/stable/api.html#agglomerative-clustering

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)

model = cuAgglomerativeClustering()
mach = machine(model, x)
fit!(mach)
preds = mach.report.labels #AgglomerativeClustering does not have a predict method
```
"""
MLJModelInterface.@mlj_model mutable struct cuAgglomerativeClustering <: MMI.Unsupervised
    handle = nothing
    verbose::Bool = false
    affinity::String = "euclidean"::(_ in ("euclidean", "l1", "l2", "manhattan", "cosine"))
    # linkage::String = "single"::(_ in ("single", ))
    # only single linkage is supported. Error when specifying linkage
    n_neighbors::Int = 15::(_ > 0)
    connectivity::String = "knn"::(_ in ("knn", "pairwise"))
end

"""
RAPIDS AI Docs for HDBSCAN: https://docs.rapids.ai/api/cuml/stable/api.html#hdbscan

Example:
```
using RAPIDS
using MLJ

x = rand(100, 5)

model = cuHDBSCAN()
mach = machine(model, x)
fit!(mach)
preds = mach.report.labels #AgglomerativeClustering does not have a predict method
```
"""
MLJModelInterface.@mlj_model mutable struct cuHDBSCAN <: MMI.Unsupervised
    handle = nothing
    alpha::Float64 = 1.0::(_ > 0)
    verbose::Bool = false
    min_cluster_size::Int = 5::(_ > 0)
    min_samples = nothing
    cluster_selection_epsilon::Float64 = 0.0::(_ >= 0)
    max_cluster_size::Int = 0::(_ >= 0)
    # TODO: Why are we getting an affinity error when specifying this parameter
    # metric = "minkowski"
    p::Int = 2::(_ > 0)
    cluster_selection_method::String = "eom"::(_ in ("eom", "leaf"))
    allow_single_cluster::Bool = false
    # requires `hdbscan` cpu python package
    # gen_min_span_tree::Bool = false
    # gen_condensed_tree::Bool = false 
    # gen_single_linkage_tree_::Bool = false
end


# Multiple dispatch for initializing models
model_init(mlj_model::cuKMeans) = cuml.KMeans(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::cuDBSCAN) = cuml.DBSCAN(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::cuAgglomerativeClustering) = cuml.AgglomerativeClustering(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::cuHDBSCAN) = cuml.HDBSCAN(; mlj_to_kwargs(mlj_model)...)


# add metadata
MMI.metadata_model(cuKMeans,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's KMeans: https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering",
	load_path    = "RAPIDS.cuKMeans"
)
MMI.metadata_model(cuDBSCAN,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's cuDBSCAN: https://docs.rapids.ai/api/cuml/stable/api.html#dbscan",
	load_path    = "RAPIDS.cuDBSCAN"
)
MMI.metadata_model(cuAgglomerativeClustering,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's Agglomerative Clustering: https://docs.rapids.ai/api/cuml/stable/api.html#agglomerative-clustering",
	load_path    = "RAPIDS.cuAgglomerativeClustering"
)
MMI.metadata_model(cuHDBSCAN,
    input_scitype   = AbstractMatrix,  # what input data is supported?
    output_scitype  = AbstractVector,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's HDBSCAN Clustering: https://docs.rapids.ai/api/cuml/stable/api.html#hdbscan",
	load_path    = "RAPIDS.cuHDBSCAN"
)

const CUML_CLUSTERING = Union{cuKMeans, cuDBSCAN, cuAgglomerativeClustering, cuHDBSCAN}

# fit methods
function MMI.fit(mlj_model::cuKMeans, verbosity, X, w=nothing)
    # initialize model, prepare data
    model = model_init(mlj_model)

    # fit the model 
    # TODO: why do we have to specify numpy array?
    model.fit(prepare_x(X))
    fitresult = (model, )

    # save result
    cache = nothing
    report = (n_iter = pyconvert(Int64, model.n_iter_), 
            labels = pyconvert(Vector{Int64}, model.labels_),
            cluster_centers = pyconvert(Matrix{Float32}, model.cluster_centers_)
    )
    return (fitresult, cache, report)
end

function MMI.fit(mlj_model::cuDBSCAN, verbosity, X, w=nothing)
    # initialize model, prepare data
    model = model_init(mlj_model)

    # fit the model
    py_preds = model.fit_predict(prepare_x(X))
    fitresult = (model, py_preds)

    # save result
    cache = nothing
    report = (n_features_in = pyconvert(Int, model.n_features_in_), 
            labels = pyconvert(Vector{Int}, model.labels_)
    )
    return (fitresult, cache, report)
end

function MMI.fit(mlj_model::cuAgglomerativeClustering, verbosity, X, w=nothing)
    # initialize model, prepare data
    model = model_init(mlj_model)
    X = MMI.matrix(X) .|> Float32

    # fit the model
    model.fit(prepare_x(X))
    fitresult = (model, )

    # save result
    cache = nothing
    report = (children = pyconvert(Matrix{Int}, model.children_), 
            labels = pyconvert(Vector{Int}, model.labels_),
    )
    return (fitresult, cache, report)
end

function MMI.fit(mlj_model::cuHDBSCAN, verbosity, X, w=nothing)
    # initialize model, prepare data
    model = model_init(mlj_model)
    X = MMI.matrix(X) .|> Float32

    # fit the model
    model.fit(prepare_x(X))
    fitresult = (model, )

    # save result
    cache = nothing
    report = (children = pyconvert(Matrix{Float64}, model.children_), 
            labels = pyconvert(Vector{Int}, model.labels_),
            cluster_persistence = pyconvert(Matrix, numpy.array(model.cluster_persistence_)),
    )
    return (fitresult, cache, report)
end


# predict methods
function MMI.predict(mlj_model::cuKMeans, fitresult, Xnew)
    model,  = fitresult
    py_preds = model.predict(prepare_x(Xnew))
    preds = pyconvert(Vector{Int}, py_preds) 

    return preds
end
    
function MMI.predict(mlj_model::cuDBSCAN, fitresult, Xnew)
    @error "DBSCAN does not support predictions on new observations."
end

function MMI.predict(mlj_model::cuAgglomerativeClustering, fitresult, Xnew)
    @error "AgglomerativeClustering does not support predictions on new observations."
end
    
function MMI.predict(mlj_model::cuHDBSCAN, fitresult, Xnew)
    @error "HDBSCAN does not support predictions on new observations."
end

# Clustering metadata
MMI.metadata_pkg.((cuKMeans, cuDBSCAN, cuAgglomerativeClustering, cuHDBSCAN),
    name = "cuML Clustering Methods",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url  = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,      # does it wrap around some other package?
)