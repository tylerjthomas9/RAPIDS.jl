
"""
$(MMI.doc_header(PCA))

`PCA` is a wrapper for the RAPIDS PCA.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `copy=false`: If True, then copies data then removes mean from data.
- `iterated_power=15`: Used in Jacobi solver. The more iterations, the more accurate, but slower.
- `n_components=nothing`: The number of top K singular vectors / values you want.
- `random_state=nothing`: Seed for the random number generator.
- `svd_solver="full`: 
    - `full`: eigendecomposition of the covariance matrix then discards components.
    - `jacobi`: much faster as it iteratively corrects, but is less accurate.
- `tol=1e-7`: Convergence tolerance for `jacobi`. 
- `whiten=false`: If True, de-correlates the components.
- `verbose=false`: Sets logging level.


# Operations

- `tansform(mach, Xnew)`

- `inverse_transform(mach, Xtrans)`

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:


# Examples
```
using RAPIDS
using MLJBase

X = rand(100, 5)

model = PCA(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)
inverse_transform(mach, X)

println(mach.fitresult.components_)
```
"""
PCA

"""
$(MMI.doc_header(IncrementalPCA))

`IncrementalPCA` is a wrapper for the RAPIDS IncrementalPCA.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `copy=false`: If True, then copies data then removes mean from data.
- `n_components=nothing`: The number of top K singular vectors / values you want.
- `tol=1e-7`: Convergence tolerance for `jacobi`. 
- `whiten=false`: If True, de-correlates the components.
- `batch_size=nothing`: The number of samples to use for each batch. Only used when calling `fit`.
- `verbose=false`: Sets logging level.


# Operations

- `tansform(mach, Xnew)`

- `inverse_transform(mach, Xtrans)`

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:


# Examples
```
using RAPIDS
using MLJBase

X = rand(100, 5)

model = IncrementalPCA(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)
inverse_transform(mach, X)

println(mach.fitresult.components_)
```
"""
IncrementalPCA

"""
$(MMI.doc_header(TruncatedSVD))

`TruncatedSVD` is a wrapper for the RAPIDS TruncatedSVD.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `n_components=nothing`: The number of top K singular vectors / values you want.
- `n_iter=15`: The number of top K singular vectors / values you want.
- `random_state=nothing`: Seed for the random number generator.
- `tol=1e-7`: Convergence tolerance for `jacobi`.
- `verbose=false`: Sets logging level.


# Operations

- `tansform(mach, Xnew)`

- `inverse_transform(mach, Xtrans)`

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:


# Examples
```
using RAPIDS
using MLJBase

X = rand(100, 5)

model = TruncatedSVD(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)
inverse_transform(mach, X_trans)

println(mach.fitresult.components_)
```
"""
TruncatedSVD



"""
$(MMI.doc_header(UMAP))

`UMAP` is a wrapper for the RAPIDS UMAP.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `n_neighbors=15`: The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
- `n_components=2`: The dimension of the space to embed into.
- `metric="euclidean"`: `l1`, `cityblock`, `taxicab`, `manhattan`, `euclidean`, `l2`, `sqeuclidean`, `canberra`, `minkowski`, `chebyshev`, `linf`, `cosine`, `correlation`, `hellinger`, `hamming`, `jaccard`
- `n_epochs=nothing`: The number of training epochs to be used in optimizing the low dimensional embedding. 
- `learning_rate=1.0`: The initial learning rate for the embedding optimization.
- `init="spectral"`: How to initialize the low dimensional embedding. 
    - `spectral`: use a spectral embedding of the fuzzy 1-skeleton.
    - `random`: assign initial embedding positions at random.
- `min_dist=0.1`: The effective minimum distance between embedded points.
- `spread=1.0`: The effective scale of embedded points.
- `set_op_mix_ratio=1.0`: Interpolate between (fuzzy) union and intersection as the set operation used to combine local fuzzy simplicial sets to obtain a global fuzzy simplicial sets.
- `local_connectivity=1`: The local connectivity required - i.e. the number of nearest neighbors that should be assumed to be connected at a local level. 
- `repulsion_strength=1.0`: Weighting applied to negative samples in low dimensional embedding optimization.
- `negative_sample_rate=5`: The number of negative samples to select per positive sample in the optimization process.
- `transform_queue_size=4.0`: For transform operations (embedding new points using a trained model this will control how aggressively to search for nearest neighbors.
- `a=nothing`: More specific parameters controlling the embedding.
- `b=nothing`: More specific parameters controlling the embedding.
- `hash_input=false`: Hash input, so exact embeddings are return when transform is called on the same data upon which the model was trained.
- `random_state=nothing`: Seed for the random number generator.`
- `callback=nothing`: Used to intercept the internal state of embeddings while they are being trained.
- `verbose=false`: Sets logging level.


# Operations

- `tansform(mach, Xnew)`

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:


# Examples
```
using RAPIDS
using MLJBase

X = rand(100, 5)

model = UMAP(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)

println(mach.fitresult.embedding_)
```
"""
UMAP


"""
$(MMI.doc_header(GaussianRandomProjection))

`GaussianRandomProjection` is a wrapper for the RAPIDS GaussianRandomProjection.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `n_components="auto"`: Dimensionality of the target projection space. 
- `eps=0.1`: Error tolerance during projection.
- `random_state=nothing`: Seed for the random number generator.
- `verbose=false`: Sets logging level.


# Operations

- `tansform(mach, Xnew)`

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:


# Examples
```
using RAPIDS
using MLJBase

X = rand(100, 5)

model = GaussianRandomProjection(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)
```
"""
GaussianRandomProjection


"""
$(MMI.doc_header(SparseRandomProjection))

`SparseRandomProjection` is a wrapper for the RAPIDS SparseRandomProjection.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `n_components=2`: Dimensionality of the target projection space. 
- `density="auto"`: Ratio of non-zero component in the random projection matrix.
- `eps=0.1`: Error tolerance during projection.
- `random_state=nothing`: Seed for the random number generator.
- `verbose=false`: Sets logging level.


# Operations

- `tansform(mach, Xnew)`

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:


# Examples
```
using RAPIDS
using MLJBase

X = rand(100, 5)

model = SparseRandomProjection(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)
```
"""
SparseRandomProjection

"""
$(MMI.doc_header(TSNE))

`TSNE` is a wrapper for the RAPIDS TSNE.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `n_components=2`: The output dimensionality size. Currently only 2 is supported.
- `perplexity=30.0`
- `early_exaggeration=12.0`: Space between clusters.
- `late_exaggeration=1.0`: Space between clusters. It may be beneficial to increase this slightly to improve cluster separation.
- `learning_rate=200.0`: The learning rate usually between (10, 1000).
- `n_iter`: Number of epochs. The more epochs, the more stable/accurate the final embedding.
- `n_iter_without_progress=300`: Currently unused. When the KL Divergence becomes too small after some iterations, terminate t-SNE early.
- `min_grad_norm=1e-7`: The minimum gradient norm for when t-SNE will terminate early. Used in the `exact` and `fft` algorithms.
- `metric="euclidean"`: `l1`, `cityblock`, `manhattan`, `euclidean`, `l2`, `sqeuclidean`, `minkowski`, `chebyshev`, `cosine`, `correlation`
- `init="random"`: Only `random` is supported.
- `method="fft"`: `barnes_hut` and `fft` are fast approximations. `exact` is more accurate but slower.
- `angle=0.5`: Valid values are between 0.0 and 1.0, which trade off speed and accuracy, respectively.
- `learning_rate_method="adaptive"`: `adaptive` or `none`.
- `n_neighbors=90`: The number of datapoints you want to use in the attractive forces.
- `perplexity_max_iter=100`: The number of epochs the best gaussian bands are found for.
- `exaggeration_iter=250`: To promote the growth of clusters, set this higher.
- `pre_momentum=0.5`: During the exaggeration iteration, more forcefully apply gradients.
- `post_momentum=0.8`: During the late phases, less forcefully apply gradients.
- `square_distances=true`: Whether TSNE should square the distance values.
- `random_state=nothing`: Seed for the random number generator.
- `verbose=false`: Sets logging level.


# Operations

- `tansform(mach, Xnew)`

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `model`: the trained model object created by the RAPIDS.jl package

# Report

The fields of `report(mach)` are:


# Examples
```
using RAPIDS
using MLJBase

X = rand(100, 5)

model = TSNE(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)

println(mach.fitresult.kl_divergence_)
```
"""
TSNE
