
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
MMI.load_path(::Type{<:PCA}) = "$PKG.CuML.PCA"
MMI.load_path(::Type{<:IncrementalPCA}) = "$PKG.CuML.IncrementalPCA"
MMI.load_path(::Type{<:TruncatedSVD}) = "$PKG.CuML.TruncatedSVD"
MMI.load_path(::Type{<:UMAP}) = "$PKG.CuML.UMAP"
MMI.load_path(::Type{<:GaussianRandomProjection}) = "$PKG.CuML.GaussianRandomProjection"
MMI.load_path(::Type{<:SparseRandomProjection}) = "$PKG.CuML.SparseRandomProjection"
MMI.load_path(::Type{<:TSNE}) = "$PKG.CuML.TSNE"

function MMI.input_scitype(::Type{<:CUML_DIMENSIONALITY_REDUCTION})
    return Union{
        Table{<:Union{AbstractVector{<:Continuous}, AbstractVector{<:Count}, AbstractVector{<:OrderedFactor}, AbstractVector{<:Multiclass}}},
        AbstractMatrix{MMI.Continuous},
    }
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
    model.fit(X)
    fitresult = model

    # save result
    cache = nothing
    report = ()
    return (fitresult, cache, report)
end

# transform methods
function MMI.transform(
    mlj_model::Union{
        PCA,IncrementalPCA,UMAP,TruncatedSVD,SparseRandomProjection,GaussianRandomProjection
    },
    fitresult,
    Xnew,
)
    model = fitresult
    py_preds = model.transform(Xnew)
    preds = pyconvert(Array, py_preds)

    return preds
end

# todo figure out how to handle TSNE's lack of transform
# right now we have to refit the model when transforming
function MMI.transform(mlj_model::TSNE, fitresult, Xnew)
    model = fitresult
    py_preds = model.fit_transform(Xnew)
    preds = pyconvert(Array, py_preds)

    return preds
end

function MMI.inverse_transform(mlj_model::Union{PCA,TruncatedSVD}, fitresult, Xnew)
    model = fitresult
    py_preds = model.inverse_transform(Xnew)
    preds = pyconvert(Array, py_preds)

    return preds
end

# Clustering metadata
MMI.metadata_pkg.(
    (PCA, IncrementalPCA, TruncatedSVD, UMAP, GaussianRandomProjection, TSNE),
    name="cuML Dimensionality Reduction and Manifold Learning Methods",
    uuid="2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url="https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia=false,          # is it written entirely in Julia?
    license="MIT",        # your package license
    is_wrapper=true,
)
