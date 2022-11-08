
# Model hyperparameters

MMI.@mlj_model mutable struct KMeans <: MMI.Unsupervised
    n_clusters::Int64 = 8::(_ > 0)
    max_iter::Int64 = 300::(_ > 0)
    tol::Float64 = 1e-4::(_ > 0)
    verbose::Bool = false
    random_state::Int = 1::(_ > 0)
    init::String = "scalable-k-means++"::(_ in
                                          ("scalable-k-means++", "k-means||", "random"))
    n_init::Int = 1::(_ > 0)
    oversampling_factor::Float64 = 2.0::(_ > 0)
    max_samples_per_batch::Int64 = 32768::(_ > 0)
end


MMI.@mlj_model mutable struct DBSCAN <: MMI.Unsupervised
    eps::Float64 = 1e-4::(_ > 0)
    min_samples::Int = 1::(_ > 0)
    metric::String = "euclidean"::(_ in ("euclidean", "precomputed"))
    verbose::Bool = false
    max_mbytes_per_batch::Union{Nothing, Int} = nothing
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
    min_samples::Union{Nothing, Int} = nothing
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
MMI.load_path(::Type{<:KMeans}) = "$PKG.KMeans"
MMI.load_path(::Type{<:DBSCAN}) = "$PKG.DBSCAN"
MMI.load_path(::Type{<:AgglomerativeClustering}) = "$PKG.AgglomerativeClustering"
MMI.load_path(::Type{<:HDBSCAN}) = "$PKG.HDBSCAN"

function MMI.input_scitype(::Type{<:CUML_CLUSTERING})
    return Union{AbstractMatrix{<:Continuous},Table(Continuous)}
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
function MMI.fit(mlj_model::CUML_CLUSTERING, verbosity, X, w=nothing)
    X_numpy = prepare_input(X)

    # fit the model
    model = model_init(mlj_model)
    model.fit(X_numpy)
    fitresult = model

    # save result
    cache = nothing
    labels = pyconvert(Vector, fitresult.labels_) |> MMI.categorical
    report = (features = _feature_names(X),
            lablels = labels)
    return (fitresult, cache, report)
end

# predict methods
function MMI.predict(mlj_model::KMeans, fitresult, Xnew)
    model = fitresult
    py_preds = model.predict(prepare_input(Xnew))
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
MMI.metadata_pkg.((KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN),
                  name="cuML Clustering Methods",
                  uuid="2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
                  url="https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
                  julia=false,          # is it written entirely in Julia?
                  license="MIT",        # your package license
                  is_wrapper=true)



"""
$(MMI.doc_header(KMeans))

`KMeans` is a wrapper for the RAPIDS KMeans Clustering.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `n_clusters=8`: The number of clusters/centroids.
- `max_iter=300`: Maximum iterations of the EM algorithm. 
- `tol=1e-4`: Stopping criterion when centroid means do not change much.
- `random_state=1`: Seed for the random number generator.
- `init="scalable-k-means++"`
    - `scalable-k-means++` or `k-means||`: Uses fast and stable scalable kmeans++ initialization.
    - `random`: Choose `n_cluster` observations (rows) at random from data for the initial centroids.
- `n_init=1`: Number of instances the k-means algorithm will be called with different seeds. The final results will be from the instance that produces lowest inertia out of n_init instances.
- `oversampling_factor=20`: The amount of points to sample in scalable k-means++ initialization for potential centroids.
- `max_samples_per_batch=32768`: The number of data samples to use for batches of the pairwise distance computation.
- `verbose=false`: Sets logging level.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are class assignments. 


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:

- `features`: the names of the features encountered in training.

- `labels`: Vector of observation labels. 

# Examples
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
KMeans

"""
$(MMI.doc_header(DBSCAN))

`DBSCAN` is a wrapper for the RAPIDS DBSCAN Clustering.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `eps=0.5`: The maximum distance between 2 points such they reside in the same neighborhood.
- `min_samples=5`: The number of samples in a neighborhood such that this group can be considered as an important core point (including the point itself).
- `metric="euclidean`: The metric to use when calculating distances between points.
    - `euclidean`, `cosine`, `precomputed`
- `max_mbytes_per_batch=nothing`: Calculate batch size using no more than this number of megabytes for the pairwise distance computation. 
- `verbose=false`: Sets logging level.


# Operations


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:

- `features`: the names of the features encountered in training.

- `labels`: Vector of observation labels. 

# Examples
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
DBSCAN

"""
$(MMI.doc_header(AgglomerativeClustering))

`AgglomerativeClustering` is a wrapper for the RAPIDS Agglomerative Clustering.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `n_clusters=8`: The number of clusters.
- `affinity="euclidean"`: Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, or “cosine”. 
- `n_neighbors=15`: The number of neighbors to compute when `connectivity = “knn”`
- `connectivity="knn"`:
    - `knn` will sparsify the fully-connected connectivity matrix to save memory and enable much larger inputs.
    - `pairwise`  will compute the entire fully-connected graph of pairwise distances between each set of points.
- `verbose=false`: Sets logging level.


# Operations


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:

- `features`: the names of the features encountered in training.

- `labels`: Vector of observation labels. 

# Examples
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
AgglomerativeClustering

"""
$(MMI.doc_header(HDBSCAN))

`HDBSCAN` is a wrapper for the RAPIDS HDBSCAN Clustering.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `alpha=1.0`: A distance scaling parameter as used in robust single linkage.
- `min_cluster_size=5`: The minimum number of samples in a group for that group to be considered a cluster.
- `min_samples=nothing`: The number of samples in a neighborhood for a point to be considered as a core point.
- `cluster_selection_epsilon=0.0`: A distance threshold. Clusters below this value will be merged.
- `max_cluster_size=0`: A limit to the size of clusters returned by the eom algorithm.
- `p=2`: p value to use if using the minkowski metric.
- `cluster_selection_method="eom"`: The method used to select clusters from the condensed tree. `eom`/`leaf`
- `allow_single_cluster=false`: Allow `HDBSCAN` to produce a single cluster.
- `verbose=false`: Sets logging level.


# Operations


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:

- `features`: the names of the features encountered in training.

- `labels`: Vector of observation labels. 

# Examples
```
using RAPIDS
using MLJ

X = rand(100, 5)

model = HDBSCAN()
mach = machine(model, X)
fit!(mach)
preds = mach.report.labels #HDBSCAN does not have a predict method
```
"""
HDBSCAN
