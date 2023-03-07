
# Model hyperparameters

MMI.@mlj_model mutable struct KMeans <: MMI.Unsupervised
    n_clusters::Int64 = 8::(_ > 0)
    max_iter::Int64 = 300::(_ > 0)
    tol::Float64 = 1e-4::(_ > 0)
    verbose::Bool = false
    random_state::Int = 1::(_ > 0)
    init::String =
        "scalable-k-means++"::(_ in ("scalable-k-means++", "k-means||", "random"))
    n_init::Int = 1::(_ > 0)
    oversampling_factor::Float64 = 2.0::(_ > 0)
    max_samples_per_batch::Int64 = 32768::(_ > 0)
end

MMI.@mlj_model mutable struct DBSCAN <: MMI.Unsupervised
    eps::Float64 = 1e-4::(_ > 0)
    min_samples::Int = 1::(_ > 0)
    metric::String = "euclidean"::(_ in ("euclidean", "precomputed"))
    verbose::Bool = false
    max_mbytes_per_batch::Union{Nothing,Int} = nothing
    calc_core_sample_indices::Bool = true
end

MMI.@mlj_model mutable struct AgglomerativeClustering <: MMI.Unsupervised
    verbose::Bool = false
    affinity::String = "euclidean"::(_ in ("euclidean", "l1", "l2", "manhattan", "cosine"))
    # linkage::String = "single"::(_ in ("single", ))
    # only single linkage is supported. Error when specifying linkage
    n_neighbors::Int = 15::(_ > 0)
    connectivity::String = "knn"::(_ in ("knn", "pairwise"))
end

MMI.@mlj_model mutable struct HDBSCAN <: MMI.Unsupervised
    alpha::Float64 = 1.0::(_ > 0)
    verbose::Bool = false
    min_cluster_size::Int = 5::(_ > 0)
    min_samples::Union{Nothing,Int} = nothing
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
function model_init(mlj_model::AgglomerativeClustering)
    return cuml.AgglomerativeClustering(; mlj_to_kwargs(mlj_model)...)
end
model_init(mlj_model::HDBSCAN) = cuml.HDBSCAN(; mlj_to_kwargs(mlj_model)...)

const CUML_CLUSTERING = Union{KMeans,DBSCAN,AgglomerativeClustering,HDBSCAN}

# add metadata
MMI.load_path(::Type{<:KMeans}) = "$PKG.CuML.KMeans"
MMI.load_path(::Type{<:DBSCAN}) = "$PKG.CuML.DBSCAN"
MMI.load_path(::Type{<:AgglomerativeClustering}) = "$PKG.CuML.AgglomerativeClustering"
MMI.load_path(::Type{<:HDBSCAN}) = "$PKG.CuML.HDBSCAN"

function MMI.input_scitype(::Type{<:CUML_CLUSTERING})
    return Union{AbstractMatrix{<:MMI.Continuous},Table(MMI.Continuous)}
end

function MMI.docstring(::Type{<:KMeans})
    return "cuML's KMeans: https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering"
end
function MMI.docstring(::Type{<:DBSCAN})
    return "cuML's DBSCAN: https://docs.rapids.ai/api/cuml/stable/api.html#dbscan"
end
function MMI.docstring(::Type{<:AgglomerativeClustering})
    return "cuML's AgglomerativeClustering: https://docs.rapids.ai/api/cuml/stable/api.html#agglomerative-clustering"
end
function MMI.docstring(::Type{<:HDBSCAN})
    return "cuML's HDBSCAN: https://docs.rapids.ai/api/cuml/stable/api.html#hdbscan"
end

# fit methods
function MMI.fit(mlj_model::CUML_CLUSTERING, verbosity, X, w = nothing)
    X_numpy = to_numpy(X)

    # fit the model
    model = model_init(mlj_model)
    model.fit(X_numpy)
    fitresult = model

    # save result
    cache = nothing
    labels = MMI.categorical(pyconvert(Vector, fitresult.labels_))
    report = (features = _feature_names(X), labels = labels)
    return (fitresult, cache, report)
end

# predict methods
function MMI.predict(mlj_model::KMeans, fitresult, Xnew)
    model = fitresult
    py_preds = model.predict(to_numpy(Xnew))
    preds = MMI.categorical(pyconvert(Array, py_preds))

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
MMI.metadata_pkg.(
    (KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN),
    name = "cuML Clustering Methods",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,
)
