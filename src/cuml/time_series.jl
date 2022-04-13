

# Model hyperparameters

"""
RAPIDS Docs for Holt Winters Exponential Smoothing: 
    https://docs.rapids.ai/api/cuml/stable/api.html#holtwinters

Example:
```
using RAPIDS
using MLJ

X = [1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12,
    2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13,
    3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14] 

model = ExponentialSmoothing()
mach = machine(model, X)
fit!(mach)
forecast(mach, 4)
```
"""
MLJModelInterface.@mlj_model mutable struct ExponentialSmoothing <: MMI.Supervised
    handle = nothing
    endog = nothing
    seasonal::String = "additive"::(_ in ("add", "additive", "mul", "multiplicative"))
    seasonal_periods::Int = 2::(_ >= 0)
    start_periods::Int = 2::(_ >= 0)
    ts_num::Int = 1::(_ >= 0)
    eps::Float64 = 2.24e-3::(_ > 0)
    verbose::Bool = false
end



# Multiple dispatch for initializing models
model_init(X, mlj_model::ExponentialSmoothing) = cuml.ExponentialSmoothing(X; mlj_to_kwargs(mlj_model)...)

const CUML_TIME_SERIES = Union{ExponentialSmoothing, }


# add metadata
MMI.load_path(::Type{<:ExponentialSmoothing}) = "$PKG.ExponentialSmoothing"
MMI.input_scitype(::Type{<:CUML_TIME_SERIES}) = Union{AbstractMatrix, Table(Continuous)}
MMI.docstring(::Type{<:ExponentialSmoothing}) = "cuML's ExponentialSmoothing: https://docs.rapids.ai/api/cuml/stable/api.html#holtwinters"


# fit methods
function MMI.fit(mlj_model::CUML_TIME_SERIES, verbosity, X, w=nothing)
    # fit the model
    model = model_init(prepare_input(X), mlj_model)
    model.fit()
    fitresult = model

    # save result
    cache = nothing
    report = ()
    return (fitresult, cache, report)
end
# predict methods
# TODO: Figure out how to handle forecast with MMI
forecast(mach, steps) = pyconvert(Vector{Float32}, mach.fitresult.forecast(steps).to_numpy())

# Classification metadata
MMI.metadata_pkg.((ExponentialSmoothing),
    name = "cuML Time Series Methods",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url  = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,      # does it wrap around some other package?
)