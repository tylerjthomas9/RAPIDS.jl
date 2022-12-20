
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
using RAPIDS.CuML
using MLJBase

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
using RAPIDS.CuML
using MLJBase

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
    - `pairwise` will compute the entire fully-connected graph of pairwise distances between each set of points.
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
using RAPIDS.CuML
using MLJBase

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
using RAPIDS.CuML
using MLJBase

X = rand(100, 5)

model = HDBSCAN()
mach = machine(model, X)
fit!(mach)
preds = mach.report.labels #HDBSCAN does not have a predict method
```
"""
HDBSCAN
