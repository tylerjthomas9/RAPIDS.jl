
# Model hyperparameters
MMI.@mlj_model mutable struct PCA <: MMI.Unsupervised
    copy::Bool = false # we are passing a numpy array, so modifying the data does not matter
    iterated_power::Int = 15::(_ > 0)
    n_components::Union{Nothing,Int} = nothing
    random_state::Union{Nothing,Int} = nothing
    svd_solver::String = "full"::(_ in ("auto", "full", "jacobi"))
    tol::Float64 = 1e-7::(_ > 0)
    whiten::Bool = false
    verbose::Bool = false
end

MMI.@mlj_model mutable struct IncrementalPCA <: MMI.Unsupervised
    copy::Bool = false # we are passing a numpy array, so modifying the data does not matter
    whiten::Bool = false
    n_components::Union{Nothing,Int} = nothing
    batch_size::Union{Nothing,Int} = nothing
    verbose::Bool = false
end

MMI.@mlj_model mutable struct TruncatedSVD <: MMI.Unsupervised
    n_components = nothing
    n_iter::Int = 15::(_ > 0)
    random_state = nothing
    tol::Float64 = 1e-7::(_ > 0)
    verbose::Bool = false
end

MMI.@mlj_model mutable struct UMAP <: MMI.Unsupervised
    n_neighbors::Int = 15
    n_components::Int = 2
    metric::String = "euclidean"
    n_epochs::Union{Nothing,Int} = nothing
    learning_rate::Float64 = 1.0::(_ > 0)
    init::String = "spectral"::(_ in ("random", "spectral"))
    min_dist::Float64 = 0.1::(_ > 0)
    spread::Float64 = 1.0::(_ > 0)
    set_op_mix_ratio::Float64 = 1.0::(_ >= 0 && _ <= 1)
    local_connectivity::Int = 1::(_ > 0)
    repulsion_strength::Float64 = 1.0::(_ >= 0)
    negative_sample_rate::Int = 5::(_ >= 0)
    transform_queue_size::Float64 = 4.0::(_ >= 0)
    a::Union{Nothing,Float64} = nothing
    b::Union{Nothing,Float64} = nothing
    hash_input::Bool = false
    random_state::Union{Nothing,Int} = nothing
    callback::Union{Nothing,Py} = nothing
    verbose::Bool = false
end

MMI.@mlj_model mutable struct GaussianRandomProjection <: MMI.Unsupervised
    n_components::Union{String,Int} = "auto"
    eps::Float64 = 0.1::(_ > 0)
    random_state::Union{Nothing,Int} = nothing
    verbose::Bool = false
end

MMI.@mlj_model mutable struct SparseRandomProjection <: MMI.Unsupervised
    n_components::Union{String,Int} = "auto"
    density::Union{String,Float64} = "auto"
    eps::Float64 = 0.1::(_ > 0)
    random_state::Union{Nothing,Int} = nothing
    verbose::Bool = false
end

MMI.@mlj_model mutable struct TSNE <: MMI.Unsupervised
    n_components::Int = 2::(_ == 2)
    perplexity::Float64 = 30.0::(_ > 0)
    early_exaggeration::Float64 = 12.0::(_ > 0)
    late_exaggeration::Float64 = 1.0::(_ > 0)
    learning_rate::Float64 = 200.0::(_ > 0)
    n_iter::Int = 1000::(_ > 0)
    n_iter_without_progress::Int = 300::(_ > 0)
    min_grad_norm::Float64 = 1e-7::(_ > 0)
    metric::String = "euclidean"::(_ in ("euclidean",))
    init::String = "random"::(_ in ("random",))
    method::String = "fft"::(_ in ("barnes_hut", "exact", "fft"))
    angle::Float64 = 0.5::(_ >= 0 && _ <= 1)
    learning_rate_method = "adaptive"
    n_neighbors::Int = 90::(_ > 0)
    perplexity_max_iter::Int = 100::(_ >= 0)
    exaggeration_iter::Int = 250::(_ >= 0)
    pre_momentum::Float64 = 0.5::(_ >= 0)
    post_momentum::Float64 = 0.8::(_ >= 0)
    square_distances::Bool = true
    random_state::Union{Nothing,Int} = nothing
    verbose::Bool = false
end

# Multiple dispatch for initializing models
model_init(mlj_model::PCA) = cuml.decomposition.PCA(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::IncrementalPCA) = cuml.IncrementalPCA(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::TruncatedSVD) = cuml.TruncatedSVD(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::UMAP) = cuml.UMAP(; mlj_to_kwargs(mlj_model)...)
function model_init(mlj_model::GaussianRandomProjection)
    return cuml.random_projection.GaussianRandomProjection(; mlj_to_kwargs(mlj_model)...)
end
function model_init(mlj_model::SparseRandomProjection)
    return cuml.random_projection.SparseRandomProjection(; mlj_to_kwargs(mlj_model)...)
end
model_init(mlj_model::TSNE) = cuml.TSNE(; mlj_to_kwargs(mlj_model)...)

const CUML_DIMENSIONALITY_REDUCTION = Union{
    PCA,
    IncrementalPCA,
    TruncatedSVD,
    UMAP,
    SparseRandomProjection,
    GaussianRandomProjection,
    TSNE,
}

# add metadata
MMI.load_path(::Type{<:PCA}) = "$PKG.PCA"
MMI.load_path(::Type{<:IncrementalPCA}) = "$PKG.IncrementalPCA"
MMI.load_path(::Type{<:TruncatedSVD}) = "$PKG.TruncatedSVD"
MMI.load_path(::Type{<:UMAP}) = "$PKG.UMAP"
MMI.load_path(::Type{<:GaussianRandomProjection}) = "$PKG.GaussianRandomProjection"
MMI.load_path(::Type{<:SparseRandomProjection}) = "$PKG.SparseRandomProjection"
MMI.load_path(::Type{<:TSNE}) = "$PKG.TSNE"

function MMI.input_scitype(::Type{<:CUML_DIMENSIONALITY_REDUCTION})
    return Union{AbstractMatrix{<:Continuous},Table(Continuous)}
end

function MMI.docstring(::Type{<:PCA})
    return "cuML's PCA: https://docs.rapids.ai/api/cuml/stable/api.html#principal-component-analysis"
end
function MMI.docstring(::Type{<:IncrementalPCA})
    return "cuML's IncrementalPCA: https://docs.rapids.ai/api/cuml/stable/api.html#incremental-pca"
end
function MMI.docstring(::Type{<:TruncatedSVD})
    return "cuML's TruncatedSVD: https://docs.rapids.ai/api/cuml/stable/api.html#truncated-svd"
end
function MMI.docstring(::Type{<:UMAP})
    return "cuML's UMAP: https://docs.rapids.ai/api/cuml/stable/api.html#umap"
end
function MMI.docstring(::Type{<:GaussianRandomProjection})
    return "cuML's GaussianRandomProjection: https://docs.rapids.ai/api/cuml/nightly/api.html#cuml.random_projection.GaussianRandomProjection"
end
function MMI.docstring(::Type{<:SparseRandomProjection})
    return "cuML's SparseRandomProjection: https://docs.rapids.ai/api/cuml/nightly/api.html#cuml.random_projection.SparseRandomProjection"
end
function MMI.docstring(::Type{<:TSNE})
    return "cuML's TSNE: https://docs.rapids.ai/api/cuml/stable/api.html#tsne"
end

function MMI.fit(mlj_model::CUML_DIMENSIONALITY_REDUCTION, verbosity, X)
    # fit model
    model = model_init(mlj_model)
    model.fit(prepare_input(X))
    fitresult = model

    # save result
    cache = nothing
    report = ()
    return (fitresult, cache, report)
end

# transform methods
function MMI.transform(
    mlj_model::Union{
        PCA,
        IncrementalPCA,
        UMAP,
        TruncatedSVD,
        SparseRandomProjection,
        GaussianRandomProjection,
    },
    fitresult,
    Xnew,
)
    model = fitresult
    py_preds = model.transform(prepare_input(Xnew))
    preds = pyconvert(Array, py_preds)

    return preds
end

# todo figure out how to handle TSNE's lack of transform
# right now we have to refit the model when transforming
function MMI.transform(mlj_model::TSNE, fitresult, Xnew)
    model = fitresult
    py_preds = model.fit_transform(prepare_input(Xnew))
    preds = pyconvert(Array, py_preds)

    return preds
end

function MMI.inverse_transform(mlj_model::Union{PCA,TruncatedSVD}, fitresult, Xnew)
    model = fitresult
    py_preds = model.inverse_transform(prepare_input(Xnew))
    preds = pyconvert(Array, py_preds)

    return preds
end

# Clustering metadata
MMI.metadata_pkg.(
    (PCA, IncrementalPCA, TruncatedSVD, UMAP, GaussianRandomProjection, TSNE),
    name = "cuML Dimensionality Reduction and Manifold Learning Methods",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,
)



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
