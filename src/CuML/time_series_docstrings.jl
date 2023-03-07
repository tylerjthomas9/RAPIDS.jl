
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
using RAPIDS.CuML
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
using RAPIDS.CuML
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
