

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



# Multiple dispatch for initializing models
model_init(mlj_model::PCA) = cuml.PCA(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::IncrementalPCA) = cuml.IncrementalPCA(; mlj_to_kwargs(mlj_model)...)

# add metadata
MMI.metadata_model(PCA,
    input_scitype   = AbstractMatrix,  
    output_scitype  = AbstractVector,  
    supports_weights = true,           
    descr = "cuML's PCA: https://docs.rapids.ai/api/cuml/stable/api.html##principal-component-analysis",
	load_path    = "RAPIDS.PCA"
)
MMI.metadata_model(IncrementalPCA,
    input_scitype   = AbstractMatrix,  
    output_scitype  = AbstractVector,  
    supports_weights = true,           
    descr = "cuML's IncrementalPCA: https://docs.rapids.ai/api/cuml/stable/api.html##incremental-pca",
	load_path    = "RAPIDS.IncrementalPCA"
)



const CUML_DIMENSIONALITY_REDUCTION = Union{PCA, IncrementalPCA}

function MMI.fit(mlj_model::CUML_DIMENSIONALITY_REDUCTION, verbosity, X, w=nothing)
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
function MMI.transform(mlj_model::CUML_DIMENSIONALITY_REDUCTION, fitresult, Xnew)
    model  = fitresult
    py_preds = model.transform(prepare_input(Xnew))
    preds = pyconvert(Array, py_preds) 

    return preds
end

function MMI.inverse_transform(mlj_model::PCA, fitresult, Xnew)
    model = fitresult
    py_preds = model.inverse_transform(prepare_input(Xnew))
    preds = pyconvert(Array, py_preds) 

    return preds
end

# Clustering metadata
MMI.metadata_pkg.((PCA, IncrementalPCA),
    name = "cuML Dimensionality Reduction and Manifold Learning Methods",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url  = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,      # does it wrap around some other package?
)