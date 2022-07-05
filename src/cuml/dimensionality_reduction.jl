

# Model hyperparameters

"""
RAPIDS Docs for PCA: 
    https://docs.rapids.ai/api/cuml/stable/api.html#principal-component-analysis

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)

model = PCA(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)
inverse_transform(mach, X)

println(mach.fitresult.components_)
```
"""
MLJModelInterface.@mlj_model mutable struct PCA <: MMI.Unsupervised
    handle = nothing
    copy::Bool = false # we are passing a numpy array, so modifying the data does not matter
    iterated_power::Int = 15::(_ > 0)
    n_components = nothing
    random_state = nothing
    svd_solver::String = "full"::(_ in ("auto", "full", "jacobi"))
    tol::Float64 = 1e-7::(_ > 0)
    verbose::Bool = false
end

"""
RAPIDS Docs for IncrementalPCA: 
    https://docs.rapids.ai/api/cuml/stable/api.html#incremental-pca

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)

model = IncrementalPCA(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)

println(mach.fitresult.components_)
```
"""
MLJModelInterface.@mlj_model mutable struct IncrementalPCA <: MMI.Unsupervised
    handle = nothing
    copy::Bool = false # we are passing a numpy array, so modifying the data does not matter
    whiten::Bool = false
    n_components = nothing
    batch_size = nothing
    verbose::Bool = false
end


"""
RAPIDS Docs for TruncatedSVD: 
    https://docs.rapids.ai/api/cuml/stable/api.html#truncated-svd

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)

model = TruncatedSVD(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)

println(mach.fitresult.components_)
```
"""
MLJModelInterface.@mlj_model mutable struct TruncatedSVD <: MMI.Unsupervised
    handle = nothing
    n_components = nothing
    n_iter::Int = 15::(_ > 0)
    random_state = nothing
    tol::Float64 = 1e-7::(_ > 0)
    verbose::Bool = false
end


"""
RAPIDS Docs for UMAP: 
    https://docs.rapids.ai/api/cuml/stable/api.html#umap

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)

model = UMAP(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)

println(mach.fitresult.embedding_)
```
"""
MLJModelInterface.@mlj_model mutable struct UMAP <: MMI.Unsupervised
    handle = nothing
    n_components = nothing
    n_epochs::Int = 15::(_ > 0)
    learning_rate::Float64 = 1.0::(_ > 0)
    init::String = "spectral"::(_ in ("random", "spectral"))
    min_dist::Float64 = 0.1::(_ > 0)
    spread::Float64 = 1.0::(_ > 0)
    set_op_mix_ratio::Float64 = 1.0::(_ >= 0 && _ <= 1)
    local_connectivity::Int = 1::(_ > 0)
    repulsion_strength::Float64 = 1.0::(_ >= 0)
    negative_sample_rate::Int = 5::(_ >= 0)
    transform_queue_size::Float64 = 4.0::(_ >= 0)
    a = nothing
    b = nothing
    hash_input::Bool = false
    random_state = nothing
    #optim_batch_size = nothing # TODO: This is in the cuML docs, but not an actual argument
    callback = nothing
    verbose::Bool = false
end

"""
RAPIDS Docs for GaussianRandomProjection: 
    https://docs.rapids.ai/api/cuml/stable/api.html#random-projection

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)

model = GaussianRandomProjection(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)
```
"""
MLJModelInterface.@mlj_model mutable struct GaussianRandomProjection <: MMI.Unsupervised
    handle = nothing
    n_components = nothing
    eps::Float64 = 0.1::(_ > 0)
    random_state = nothing
    verbose::Bool = false
end


"""
RAPIDS Docs for TSNE: 
    https://docs.rapids.ai/api/cuml/stable/api.html#tsne

Example:
```
using RAPIDS
using MLJ

X = rand(100, 5)

model = TSNE(n_components=2)
mach = machine(model, X)
fit!(mach)
X_trans = transform(mach, X)

println(mach.fitresult.kl_divergence_)
```
"""
MLJModelInterface.@mlj_model mutable struct TSNE <: MMI.Unsupervised
    handle = nothing
    n_components::Int = 2::(_ > 0)
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
    square_distances::Bool = false
    random_state = nothing
    verbose::Bool = false
end


# Multiple dispatch for initializing models
model_init(mlj_model::PCA) = cuml.PCA(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::IncrementalPCA) = cuml.IncrementalPCA(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::TruncatedSVD) = cuml.TruncatedSVD(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::UMAP) = cuml.UMAP(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::GaussianRandomProjection) =
    cuml.GaussianRandomProjection(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::TSNE) = cuml.TSNE(; mlj_to_kwargs(mlj_model)...)

const CUML_DIMENSIONALITY_REDUCTION =
    Union{PCA,IncrementalPCA,TruncatedSVD,UMAP,GaussianRandomProjection,TSNE}


# add metadata
MMI.load_path(::Type{<:PCA}) = "$PKG.PCA"
MMI.load_path(::Type{<:IncrementalPCA}) = "$PKG.IncrementalPCA"
MMI.load_path(::Type{<:TruncatedSVD}) = "$PKG.TruncatedSVD"
MMI.load_path(::Type{<:UMAP}) = "$PKG.UMAP"
MMI.load_path(::Type{<:GaussianRandomProjection}) = "$PKG.GaussianRandomProjection"
MMI.load_path(::Type{<:TSNE}) = "$PKG.TSNE"

MMI.input_scitype(::Type{<:CUML_DIMENSIONALITY_REDUCTION}) =
    Union{AbstractMatrix,Table(Continuous)}

MMI.docstring(::Type{<:PCA}) =
    "cuML's PCA: https://docs.rapids.ai/api/cuml/stable/api.html#principal-component-analysis"
MMI.docstring(::Type{<:IncrementalPCA}) =
    "cuML's IncrementalPCA: https://docs.rapids.ai/api/cuml/stable/api.html#incremental-pca"
MMI.docstring(::Type{<:TruncatedSVD}) =
    "cuML's TruncatedSVD: https://docs.rapids.ai/api/cuml/stable/api.html#truncated-svd"
MMI.docstring(::Type{<:UMAP}) =
    "cuML's UMAP: https://docs.rapids.ai/api/cuml/stable/api.html#umap"
MMI.docstring(::Type{<:GaussianRandomProjection}) =
    "cuML's GaussianRandomProjection: https://docs.rapids.ai/api/cuml/stable/api.html#random-projection"
MMI.docstring(::Type{<:TSNE}) =
    "cuML's TSNE: https://docs.rapids.ai/api/cuml/stable/api.html#tsne"



function MMI.fit(mlj_model::CUML_DIMENSIONALITY_REDUCTION, verbosity, X, w = nothing)
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
    mlj_model::Union{PCA,IncrementalPCA,UMAP,TruncatedSVD,GaussianRandomProjection},
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
    is_wrapper = true,      # does it wrap around some other package?
)
