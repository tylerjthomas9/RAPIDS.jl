

"""
RAPIDS AI Docs for KMeans: https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering
"""
MLJModelInterface.@mlj_model mutable struct cuKMeans <: MMI.Unsupervised
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


model_init(mlj_model::cuKMeans) = cuml.KMeans(; mlj_to_kwargs(mlj_model)...)

MMI.metadata_model(cuKMeans,
    input_scitype   = Matrix,  # what input data is supported?
    output_scitype  = Matrix,  # for an unsupervised, what output?
    supports_weights = false,                      # does the model support sample weights?
    descr = "cuML's KMeans: https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering",
	load_path    = "RAPIDS.cuml.KMeans"
)
