
# Model hyperparameters

MMI.@mlj_model mutable struct ExponentialSmoothing <: MMI.Unsupervised
    seasonal::String = "additive"::(_ in ("add", "additive", "mul", "multiplicative"))
    seasonal_periods::Int = 2::(_ >= 0)
    start_periods::Int = 2::(_ >= 0)
    ts_num::Int = 1::(_ >= 0) #TODO: handle more time series here
    eps::Float64 = 2.24e-3::(_ > 0)
    verbose::Bool = false
end

MMI.@mlj_model mutable struct ARIMA <: MMI.Unsupervised
    #TODO: Fix tuple in struct
    # order::Tuple{Int,Int,Int} = (1,1,1)
    # season_order::Tuple{Int,Int,Int} = (0,0,0,0)
    exog::Union{Py,Nothing} = nothing
    fit_intercept::Bool = true
    simple_differencing::Bool = true
    verbose::Bool = false
end

# Multiple dispatch for initializing models
function model_init(X, mlj_model::ExponentialSmoothing)
    return cuml.ExponentialSmoothing(X; mlj_to_kwargs(mlj_model)...)
end
model_init(X, mlj_model::ARIMA) = cuml.tsa.ARIMA(X; mlj_to_kwargs(mlj_model)...)

const CUML_TIME_SERIES = Union{ExponentialSmoothing,ARIMA}

# add metadata
MMI.load_path(::Type{<:ExponentialSmoothing}) = "$PKG.ExponentialSmoothing"
MMI.load_path(::Type{<:ARIMA}) = "$PKG.ARIMA"

function MMI.input_scitype(::Type{<:CUML_TIME_SERIES})
    return Union{AbstractMatrix{<:MMI.Continuous},Table(MMI.Continuous)}
end
function MMI.docstring(::Type{<:ExponentialSmoothing})
    return "cuML's ExponentialSmoothing: https://docs.rapids.ai/api/cuml/stable/api.html#holtwinters"
end
function MMI.docstring(::Type{<:ARIMA})
    return "cuML's ARIMA: https://docs.rapids.ai/api/cuml/nightly/api.html#cuml.tsa.ARIMA"
end

# fit methods
function MMI.fit(mlj_model::CUML_TIME_SERIES, verbosity, X, w = nothing)
    # fit the model
    model = model_init(to_numpy(X), mlj_model)
    model.fit()
    fitresult = model

    # save result
    cache = nothing
    report = ()
    return (fitresult, cache, report)
end

# predict methods
# TODO: Figure out how to handle forecast with MMI
function forecast(mach, steps)
    x = mach.fitresult.forecast(steps)
    try
        x = x.to_numpy()
        return pyconvert(Vector{Float32}, x)
    catch
        return pyconvert(Vector{Float32}, numpy.array(x).flatten())
    end
end

# Classification metadata
MMI.metadata_pkg.(
    (ExponentialSmoothing, ARIMA),
    name = "cuML Time Series Methods",
    uuid = "2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
    url = "https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
    julia = false,          # is it written entirely in Julia?
    license = "MIT",        # your package license
    is_wrapper = true,
)
