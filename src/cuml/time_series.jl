
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
    return Union{AbstractMatrix{<:Continuous},Table(Continuous)}
end
function MMI.docstring(::Type{<:ExponentialSmoothing})
    return "cuML's ExponentialSmoothing: https://docs.rapids.ai/api/cuml/stable/api.html#holtwinters"
end
function MMI.docstring(::Type{<:ARIMA})
    return "cuML's ARIMA: https://docs.rapids.ai/api/cuml/nightly/api.html#cuml.tsa.ARIMA"
end

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
MMI.metadata_pkg.((ExponentialSmoothing, ARIMA),
                  name="cuML Time Series Methods",
                  uuid="2764e59e-7dd7-4b2d-a28d-ce06411bac13", # see your Project.toml
                  url="https://github.com/tylerjthomas9/RAPIDS.jl",  # URL to your package repo
                  julia=false,          # is it written entirely in Julia?
                  license="MIT",        # your package license
                  is_wrapper=true)

"""
$(MMI.doc_header(ExponentialSmoothing))

`ExponentialSmoothing` is a wrapper for the RAPIDS HoltWinters time series analysis model.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `seasonal="additive"`: Whether the seasonal trend should be calculated additively or multiplicatively.
- `seasonal_periods=2`: The seasonality of the data (how often it repeats). For monthly data this should be 12, for weekly data, this should be 7.
- `start_periods=2`: Number of seasons to be used for seasonal seed values.
- `eps=2.24e-3`: The accuracy to which gradient descent should achieve.
- `verbose=false`: Sets logging level.

# Operations

- `forecast(mach, n_timesteps)`: return a forecasted series of `n_timesteps`

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
y = [repeat([0], 50)..., repeat([1], 50)...]

model = ExponentialSmoothing()
mach = machine(model, X)
fit!(mach)
preds = forecast(mach, 4)
```
"""
ExponentialSmoothing

"""
$(MMI.doc_header(ARIMA))

`ARIMA` is a wrapper for the RAPIDS batched ARIMA model for in- and out-of-sample time-series prediction, with support for seasonality (SARIMA).

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

where

- `X`: any table or array of input features (eg, a `DataFrame`) whose columns
    each have one of the following element scitypes: `Continuous`

# Hyper-parameters

- `order=(1,1,1)`: The ARIMA order (p, d, q) of the model.
- `seasonal_order=(0,0,0,0)`:The seasonal ARIMA order (P, D, Q, s) of the model.
- `exog=nothing`: Exogenous variables, assumed to have each time series in columns, such that variables associated with a same batch member are adjacent. 
    - This must be a PyArray
- `verbose=false`: Sets logging level.
- `fit_intercept=true`: If True, include a constant trend mu in the model.
- `simple_differencing=true`: If True, the data is differenced before being passed to the Kalman filter. 
                            -If False, differencing is part of the state-space model.
- `verbose=false`: Sets logging level.

# Operations

- `forecast(mach, n_timesteps)`: return a forecasted series of `n_timesteps`

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
y = [repeat([0], 50)..., repeat([1], 50)...]

model = ARIMA()
mach = machine(model, X)
fit!(mach)
preds = forecast(mach, 4)
```
"""
ARIMA
