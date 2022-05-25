

# Model hyperparameters

"""
RAPIDS Docs for KMeans: https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)

model = KMeans()
mach = machine(model, X)
fit!(mach)
preds = predict(mach, X)
```
"""
MLJModelInterface.@mlj_model mutable struct KMeans <: MMI.Unsupervised
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
RAPIDS Docs for DBSCAN: https://docs.rapids.ai/api/cuml/stable/api.html#dbscan


Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)

model = DBSCAN()
mach = machine(model, X)
fit!(mach)
preds = mach.report.labels #DBSCAN does not have a predict method
```
"""
MLJModelInterface.@mlj_model mutable struct DBSCAN <: MMI.Unsupervised
    handle = nothing
    eps::Float64 = 1e-4::(_ > 0)
    min_samples::Int = 1::(_ > 0)
    metric::String = "euclidean"::(_ in ("euclidean","precomputed"))
    verbose::Bool = false
    max_mbytes_per_batch = nothing
    calc_core_sample_indices::Bool = true
end

"""
RAPIDS Docs for AgglomerativeClustering: https://docs.rapids.ai/api/cuml/stable/api.html#agglomerative-clustering

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)

model = AgglomerativeClustering()
mach = machine(model, X)
fit!(mach)
preds = mach.report.labels #AgglomerativeClustering does not have a predict method
```
"""
MLJModelInterface.@mlj_model mutable struct AgglomerativeClustering <: MMI.Unsupervised
    handle = nothing
    verbose::Bool = false
    affinity::String = "euclidean"::(_ in ("euclidean", "l1", "l2", "manhattan", "cosine"))
    # linkage::String = "single"::(_ in ("single", ))
    # only single linkage is supported. Error when specifying linkage
    n_neighbors::Int = 15::(_ > 0)
    connectivity::String = "knn"::(_ in ("knn", "pairwise"))
end

"""
RAPIDS Docs for HDBSCAN: https://docs.rapids.ai/api/cuml/stable/api.html#hdbscan

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)

model = HDBSCAN()
mach = machine(model, X)
fit!(mach)
preds = mach.report.labels #AgglomerativeClustering does not have a predict method
```
"""
MLJModelInterface.@mlj_model mutable struct HDBSCAN <: MMI.Unsupervised
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
model_init(mlj_model::KMeans) = cuml.KMeans(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::DBSCAN) = cuml.DBSCAN(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::AgglomerativeClustering) = cuml.AgglomerativeClustering(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::HDBSCAN) = cuml.HDBSCAN(; mlj_to_kwargs(mlj_model)...)

const CUML_CLUSTERING = Union{KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN}

# add metadata
MMI.load_path(::Type{<:KMeans}) = "$PKG.KMeans"
MMI.load_path(::Type{<:DBSCAN}) = "$PKG.DBSCAN"
MMI.load_path(::Type{<:AgglomerativeClustering}) = "$PKG.AgglomerativeClustering"
MMI.load_path(::Type{<:HDBSCAN}) = "$PKG.HDBSCAN"

MMI.input_scitype(::Type{<:CUML_CLUSTERING}) = Union{AbstractMatrix, Table(Continuous)}
MMI.target_scitype(::Type{<:KMeans}) = AbstractVector{<:Finite}

MMI.docstring(::Type{<:KMeans}) = "cuML's KMeans: https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering"
MMI.docstring(::Type{<:DBSCAN}) = "cuML's DBSCAN: https://docs.rapids.ai/api/cuml/stable/api.html#dbscan"
MMI.docstring(::Type{<:AgglomerativeClustering}) = "cuML's AgglomerativeClustering: https://docs.rapids.ai/api/cuml/stable/api.html#agglomerative-clustering"
MMI.docstring(::Type{<:HDBSCAN}) = "cuML's HDBSCAN: https://docs.rapids.ai/api/cuml/stable/api.html#hdbscan"


# fit methods
function MMI.fit(mlj_model::CUML_CLUSTERING, verbosity, X, w=nothing)
    # fit the model
    model = model_init(mlj_model)
    model.fit(prepare_input(X))
    fitresult = model

    # save result
    cache = nothing
    report = ()
    return (fitresult, cache, report)
end



# predict methods
function MMI.predict(mlj_model::KMeans, fitresult, Xnew)
    model  = fitresult
    py_preds = model.predict(prepare_input(Xnew))
    preds = pyconvert(Array, py_preds) 

    return preds
end
    
function MMI.predict(mlj_model::DBSCAN, fitresult, Xnew)
    @error "DBSCAN does not support predictions on new observations."
end

function MMI.predict(mlj_model::AgglomerativeClustering, fitresult, Xnew)
    @error "AgglomerativeClustering does not support predictions on new observations."
end
    
function MMI.predict(mlj_model::HDBSCAN, fitresult, Xnew)
    @error "HDBSCAN does not support predictions on new observations."
end

# Clustering metadata
MMI.metadata_pkg.((KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN),
    name = "cuML Clustering Methods",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url  = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,      # does it wrap around some other package?
)