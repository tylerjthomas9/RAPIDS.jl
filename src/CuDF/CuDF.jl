
module CuDF

using Dates
using PythonCall
using Lazy
using Compat
using TableTraits
using Statistics

using RAPIDS: pandas, numpy

import Base: getindex, setindex!, length, size, show, merge, convert,
             join, replace, lastindex, sum, abs, any, count,
             cumprod, cumsum, diff, filter, first, last,
             min, sort, truncate, +, -, *, /, !,
             ==, >, <, >=, <=, !=, &, |,
             keys, close, get
import Statistics: mean, std, var, cov, median, quantile

include("exports.jl")

"""
    version()

Returns the version of the underlying Python Pandas library as a VersionNumber.
"""
version() = VersionNumber(pandas.__version__)

const pre_type_map = []

abstract type PandasWrapped end

PythonCall.Py(x::PandasWrapped) = x.pyo

macro pytype(name, class)
    quote
        struct $(name) <: PandasWrapped
            pyo::Py
            $(esc(name))(pyo::Py) = new(pyo)
            function $(esc(name))(args...; kwargs...)
                pandas_method = ($class)()
                return new(pycall(pandas_method, args...; kwargs...))
            end

            function $(esc(name))(dict::Dict, args...; kwargs...)
                pandas_method = ($class)()
                dict_string = Dict(string(k) => v for (k, v) in dict)
                return new(pycall(pandas_method, pydict(dict_string), args...; kwargs...))
            end
        end

        function Base.iterate(x::$name, state...)
            res = Base.iterate(x.pyo, state...)
            if res === nothing
                return nothing
            else
                value, state = res
                return pandas_wrap(value), state
            end
        end

        push!(pre_type_map, ($class, $name))
    end
end

quot(x) = Expr(:quote, x)

function convert_datetime_series_to_julia_vector(series)
    N = length(series)
    out = Array{Dates.DateTime}(undef, N)
    for i in 1:N
        # PyCall.jl overloads the getindex method on `series` to automatically convert 
        # to a Julia date type.
        out[i] = series[i]
    end
    return out
end

function Base.Array(x::PandasWrapped)
    if typeof(x) <: Series && x.pyo.dtype == np.dtype("<M8[ns]")
        return convert_datetime_series_to_julia_vector(x)
    end
    c = np.asarray(x.pyo)
    pyconvert(Array, c)
end

function Base.values(x::PandasWrapped)
    # Check if zero-copy conversion to a Julia native type
    # is possible.
    if hasproperty(x.pyo, :dtype)
        x_kind = x.pyo.dtype.kind
        if x_kind in ["i", "u", "f", "b"]
            pyarray = convert(PyArray, x.pyo."values")
            return unsafe_wrap(Array, pyarray.data, size(pyarray))
        end
    end
    # Convert element by element otherwise
    return Array(x)
end

"""
    pandas_wrap(pyo::Py)

Wrap an instance of a Pandas python class in the Julia type which corresponds
to that class.
"""
function pandas_wrap(pyo::Py)
    #TODO: Remove hardcode, and figure out why function __init__() segfaults
    # with pre_type_map -> type_map
    type_map = Dict{Py, Any}()
    type_map[pandas.core.frame."DataFrame"] = DataFrame
    type_map[pandas.core.indexing."_iLocIndexer"] = Iloc
    type_map[pandas.core.series."Series"] = Series
    type_map[pandas.core.indexing."MultiIndex"] = MultiIndex
    type_map[pandas.core.indexes.multi."Index"] = Index
    type_map[pandas.core.indexing."_LocIndexer"] = Loc
    type_map[pandas.core.groupby."DataFrameGroupBy"] = GroupBy
    type_map[pandas.core.groupby."SeriesGroupBy"] = SeriesGroupBy
    type_map[pandas.core.window."Rolling"] = Rolling
    type_map[pandas.io.pytables.HDFStore] = HDFStore
    for (pyt, pyv) in type_map
        pyt === nothing && continue
        if pyisinstance(pyo, pyt)
            return pyv(pyo)
        end
    end
    return pyconvert(Any, pyo)
end

pandas_wrap(x::Union{AbstractArray,Tuple}) = [pandas_wrap(_) for _ in x]

pandas_wrap(pyo) = pyo

fix_arg(x::StepRange) = pyeval("slice($(x.start), $(x.start+length(x)*x.step), $(x.step))", Main)
fix_arg(x::UnitRange) = fix_arg(StepRange(x.start, 1, x.stop))
fix_arg(x::Colon) = pybuiltin("slice")(nothing, nothing, nothing)
fix_arg(x) = x

function fix_arg(x, offset)
    if offset
        fix_arg(x .- 1)
    else
        fix_arg(x)
    end
end

fix_arg(x::Colon, offset) = pybuiltin("slice")(nothing, nothing, nothing)

pyattr(class, method) = pyattr(class, method, method)

function pyattr(class, jl_method, py_method)
    quote
        function $(esc(jl_method))(pyt::$class, args...; kwargs...)
            new_args = fix_arg.(args)
            method = pyt.pyo.$(string(py_method))
            pyo = pycall(method, new_args...; kwargs...)
            return wrapped = pandas_wrap(pyo)
        end
    end
end

macro pyattr(class, method)
    return pyattr(class, method)
end

macro pyattr(class, method, orig_method)
    return pyattr(class, method, orig_method)
end

"""
    pyattr_set(types, methods...)

For each Julia type `T<:PandasWrapped` in `types` and each method `m` in `methods`,
define a new function `m(t::T, args...)` that delegates to the underlying
Py wrapped by `t`.
"""
function pyattr_set(classes, methods...)
    for class in classes
        for method in methods
            @eval @pyattr($class, $method)
        end
    end
end

macro pyasvec(class)
    index_expr = quote
        function $(esc(:getindex))(pyt::$class, args...)
            offset = should_offset(pyt, args...)
            new_args = tuple([fix_arg(arg, offset) for arg in args]...)
            new_args = (length(new_args) == 1 ? new_args[1] : new_args)
            pyo = pycall(pyt.pyo.__getitem__, Py, new_args)
            return pandas_wrap(pyo)
        end

        function $(esc(:setindex!))(pyt::$class, value, idxs...)
            offset = should_offset(pyt, idxs...)
            new_idx = [fix_arg(idx, offset) for idx in idxs]
            if length(new_idx) > 1
                pandas_wrap(pycall(pyt.pyo.__setitem__, Py, tuple(new_idx...), value))
            else
                pandas_wrap(pycall(pyt.pyo.__setitem__, Py, new_idx[1], value))
            end
        end
    end
    if class in [:Iloc, :Loc, :Ix]
        length_expr = quote
            function $(esc(:length))(x::$class)
                return x.pyo.obj.__len__() + 1
            end
        end
    else
        length_expr = quote
            function $(esc(:length))(x::$class)
                return x.pyo.__len__()
            end
        end
    end

    quote
        $index_expr
        $length_expr
        function $(esc(:lastindex))(x::$class)
            return length(x)
        end
    end
end

@pytype DataFrame () -> pandas.core.frame."DataFrame"
@pytype Iloc () -> pandas.core.indexing."_iLocIndexer"
@pytype Series () -> pandas.core.series."Series"
@pytype Ix () -> version() < VersionNumber(1) ? pandas.core.indexing."_IXIndexer" :
                 nothing
@pytype MultiIndex () -> version() < VersionNumber(1) ? pandas.core.index."MultiIndex" :
                         pandas.core.indexes.multi."MultiIndex"
@pytype Index () -> version() < VersionNumber(1) ? pandas.core.index."Index" :
                    pandas.core.indexes.multi."Index"
@pytype Loc () -> pandas.core.indexing."_LocIndexer"
@pytype GroupBy () -> pandas.core.groupby."DataFrameGroupBy"
@pytype SeriesGroupBy () -> pandas.core.groupby."SeriesGroupBy"
@pytype Rolling () -> pandas.core.window."Rolling"
@pytype HDFStore () -> pandas.io.pytables.HDFStore

@pyattr GroupBy app apply
@pyattr Rolling app apply

pyattr_set([GroupBy, SeriesGroupBy], :mean, :std, :agg, :aggregate, :median,
           :var, :ohlc, :transform, :groups, :indices, :get_group, :hist, :plot, :count)

pyattr_set([Rolling], :agg, :aggregate, :apply, :corr, :count, :cov, :kurt, :max, :mean,
           :median, :min, :ndim, :quantile, :skew, :std, :sum, :validate, :var)

@pyattr GroupBy siz size

pyattr_set([DataFrame, Series], :T, :abs, :align, :any, :argsort, :asfreq, :asof,
           :boxplot, :clip, :clip_lower, :clip_upper, :corr, :corrwith, :count, :cov,
           :cummax, :cummin, :cumprod, :cumsum, :delevel, :describe, :diff, :drop,
           :drop_duplicates, :dropna, :duplicated, :fillna, :filter, :first,
           :first_valid_index,
           :head, :hist, :idxmax, :idxmin, :iloc, :isin, :join, :last, :last_valid_index,
           :loc, :mean, :median, :min, :mode, :order, :pct_change, :pivot, :plot, :quantile,
           :rank, :reindex, :reindex_axis, :reindex_like, :rename, :reorder_levels,
           :replace, :resample, :reset_index, :sample, :select, :set_index, :shift, :skew,
           :sort, :sort_index, :sortlevel, :stack, :std, :sum, :swaplevel, :tail, :take,
           :to_clipboard, :to_csv, :to_dense, :to_dict, :to_excel, :to_gbq, :to_hdf,
           :to_html,
           :to_json, :to_latex, :to_msgpack, :to_panel, :to_pickle, :to_records, :to_sparse,
           :to_sql, :to_string, :truncate, :tz_conert, :tz_localize, :unstack, :var,
           :weekday,
           :xs, :merge, :equals, :to_parquet)
pyattr_set([DataFrame], :groupby)
pyattr_set([Series, DataFrame], :rolling)
pyattr_set([HDFStore], :put, :append, :get, :select, :info, :keys, :groups, :walk, :close)

Base.size(x::Union{Loc,Iloc,Ix}) = x.pyo.obj.shape
Base.size(df::PandasWrapped, i::Integer) = size(df)[i]
Base.size(df::PandasWrapped) = df.pyo.shape

Base.isempty(df::PandasWrapped) = df.pyo.empty
Base.empty!(df::PandasWrapped) = df.pyo.drop(df.pyo.index; inplace=true)

should_offset(::Any, args...) = false
should_offset(::Union{Iloc,Index}, args...) = true

function should_offset(s::Series, arg)
    if eltype(arg) == Int64
        if eltype(index(s)) ≠ Int64
            return true
        end
    end
    return false
end

for attr in [:index, :columns]
    @eval function $attr(x::PandasWrapped)
        return pyconvert(Array, x.pyo.$(string(attr)).values)
    end
end

@pyasvec Series
@pyasvec Loc
@pyasvec Ix
@pyasvec Iloc
@pyasvec DataFrame
@pyasvec Index
@pyasvec GroupBy
@pyasvec Rolling
@pyasvec HDFStore

Base.ndims(df::Union{DataFrame,Series}) = length(size(df))

for m in
    [:read_pickle, :read_csv, :read_gbq, :read_html, :read_json, :read_excel, :read_table,
     :save, :stats, :melt, :ewma, :concat, :pivot_table, :crosstab, :cut,
     :qcut, :get_dummies, :resample, :date_range, :to_datetime, :to_timedelta,
     :bdate_range, :period_range, :ewmstd, :ewmvar, :ewmcorr, :ewmcov, :rolling_count,
     :expanding_count, :rolling_sum, :expanding_sum, :rolling_mean, :expanding_mean,
     :rolling_median, :expanding_median, :rolling_var, :expanding_var, :rolling_std,
     :expanding_std, :rolling_min, :expanding_min, :rolling_max, :expanding_max,
     :rolling_corr, :expanding_corr, :rolling_corr_pairwise, :expanding_corr_pairwise,
     :rolling_cov, :expanding_cov, :rolling_skew, :expanding_skew, :rolling_kurt,
     :expanding_kurt, :rolling_apply, :expanding_apply, :rolling_quantile,
     :expanding_quantile, :rolling_window, :to_numeric, :read_sql, :read_sql_table,
     :read_sql_query, :read_hdf, :read_parquet]
    @eval begin
        function $m(args...; kwargs...)
            method = pandas.$(string(m))
            result = pycall(method, args...; kwargs...)
            return pandas_wrap(result)
        end
    end
end

function show(io::IO, df::PandasWrapped)
    s = df.pyo.__str__()
    return println(io, s)
end

function show(io::IO, ::MIME"text/html", df::PandasWrapped)
    obj = df.pyo
    try
        return println(io, obj.to_html())
    catch
        return show(io, df)
    end
end

function query(df::DataFrame, s::AbstractString)
    return @pyexec (df=df.pyo, s="$s") => `res=df.query(s)` => res
end

function query(df::DataFrame, e::Expr) # This whole method is a terrible hack
    s = string(e)
    for (target, repl) in [("&&", "&"), ("||", "|"), ("∈", "=="), (r"!(?!=)", "~")]
        s = replace(s, target => repl)
    end
    return query(df, s)
end

macro query(df, e)
    quote
        query($(esc(df)), $(QuoteNode(e)))
    end
end

for m in [:from_arrays, :from_tuples]
    @eval function $m(args...; kwargs...)
        f = pandas."MultiIndex"[string($(quot(m)))]
        res = pycall(f, Py, args...; kwargs...)
        return pandas_wrap(res)
    end
end

for (jl_op, py_op, py_opᵒ) in [(:+, :__add__, :__add__), (:*, :__mul__, :__mul__),
                               (:/, :__div__, :__rdiv__), (:-, :__sub__, :__rsub__),
                               (:>, :__gt__, :__lt__), (:<, :__lt__, :__gt__),
                               (:>=, :__ge__, :__le__), (:<=, :__le__, :__ge__),
                               (:&, :__and__, :__and__), (:|, :__or__, :__or__)]
    @eval begin
        function $(jl_op)(x::PandasWrapped, y)
            res = x.pyo.$(string(py_op))(y)
            return pandas_wrap(res)
        end

        function $(jl_op)(x::PandasWrapped, y::PandasWrapped)
            return invoke($(jl_op), Tuple{PandasWrapped,Any}, x, y)
        end

        function $(jl_op)(y, x::PandasWrapped)
            res = x.pyo.$(string(py_opᵒ))(y)
            return pandas_wrap(res)
        end
    end
end

# Special-case the handling of equality-testing to always consider PandasWrapped
# objects as unequal to non-wrapped objects.
(==)(x::PandasWrapped, y) = false
(==)(x, y::PandasWrapped) = false
(!=)(x::PandasWrapped, y) = true
(!=)(x, y::PandasWrapped) = true
function (==)(x::PandasWrapped, y::PandasWrapped)
    return pandas_wrap(x.pyo.__eq__(y))
end
function (!=)(x::PandasWrapped, y::PandasWrapped)
    return pandas_wrap(x.pyo.__neq__(y))
end

for op in [(:-, :__neg__)]
    @eval begin
        $(op[1])(x::PandasWrapped) = pandas_wrap(x.pyo.$(quot(op[2]))())
    end
end

function setcolumns!(df::PandasWrapped, new_columns)
    return df.pyo.__setattr__("columns", new_columns)
end

function deletecolumn!(df::DataFrame, column)
    return df.pyo.__delitem__(column)
end

name(s::Series) = s.pyo.name
name!(s::Series, name) = s.pyo.name = name

include("operators.jl")

function DataFrame(pairs::Pair...)
    return DataFrame(Dict(pairs...))
end

function index!(df::PandasWrapped, new_index)
    df.pyo.index = new_index
    return df
end

function Base.eltype(s::Series)
    dtype_map = Dict(np.dtype("int64") => Int64,
                     np.dtype("float64") => Float64,
                     np.dtype("object") => String)
    return get(dtype_map, s.pyo.dtype, Any)
end

function Base.eltype(df::DataFrame)
    types = []
    for column in columns(df)
        push!(types, eltype(df[column]))
    end
    return Tuple{types...}
end

function Base.map(f::Function, s::Series)
    if eltype(s) ∈ (Int64, Float64)
        Series([f(_) for _ in values(s)])
    else
        Series([f(_) for _ in s])
    end
end

function Base.map(x, s::Series; na_action=nothing)
    return pandas_wrap(s.pyo.map(x, na_action))
end

function Base.get(df::PandasWrapped, key, default)
    return pandas_wrap(df.pyo.get(key; default=default))
end

function Base.getindex(s::Series, c::CartesianIndex{1})
    return s[c[1]]
end

function Base.copy(df::PandasWrapped)
    return pandas_wrap(df.pyo.copy())
end

function !(df::PandasWrapped)
    return pandas_wrap(df.pyo.__neg__())
end

include("tabletraits.jl")
include("tables.jl")

function DataFrame(obj)
    y = _construct_pandas_from_iterabletable(obj)
    if y === nothing
        y = _construct_pandas_from_tables(obj)
        if y === nothing
            return invoke(DataFrame, Tuple{Vararg{Any}}, obj)
        else
            return y
        end
    else
        return y
    end
end

function has_named_attr(x::Index, s)
    return x.pyo.__contains__(Symbol(s))
end

named_index(x::DataFrame) = columns(x)
named_index(x::Series) = index(x)

function has_named_attr(x::PyIterable{Any}, s::Symbol)
    pyhasattr(x, "$s")
end

function Base.getproperty(x::Union{DataFrame,Series}, s::Symbol)
    if s == :pyo
        return getfield(x, s)
    end
    if pyhasattr(x.pyo, "$s")
        return x.pyo["$s"]
    else
        return getfield(x, s)
    end
end

end
