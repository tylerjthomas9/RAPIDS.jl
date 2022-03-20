
"""
RAPIDS AI Docs for KMeans: https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering
"""
MLJModelInterface.@mlj_model mutable struct cuKMeans <: MLJModelInterface.Deterministic
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

function fit(m::cuml.KMeans, X::Matrix{Float})
    m.fit(X)
    return nothing
end

function MLJModelInterface.fit(m::cuKMeans, verbosity, X, w=nothing)
    fit(m, X)
    cache = nothing
    n_iter = m.n_iter_
    report = (n_iter=m.n_iter_, labels=m.labels_)
    return (fitresult, cache, report)
end

# Then for each model,
MLJModelInterface.metadata_model(cuKMeans,
    input_scitype   = MLJModelInterface.Table(MLJModelInterface.Continuous),  # what input data is supported?
    target_scitype  = AbstractVector{MLJModelInterface.Continuous},           # for a supervised model, what target?
    output_scitype  = MLJModelInterface.Table(MLJModelInterface.Continuous),  # for an unsupervised, what output?
    supports_weights = false,                                                  # does the model support sample weights?
    descr   = "cuML's KMeans: https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering ",
	load_path    = "RAPIDS.cuml.KMeans"
)
