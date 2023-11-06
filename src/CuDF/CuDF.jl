
module CuDF

using RAPIDS: cudf
using PythonCall
using Tables

# I/O
read_csv(args...; kwargs...) = cudf.read_csv(args...; kwargs...)
to_csv(df::Py, args...; kwargs) = df.to_csv(args...; kwargs...)
read_text(args...; kwargs...) = cudf.read_text(args...; kwargs...)
read_json(args...; kwargs...) = cudf.read_json(args...; kwargs...)
to_json(df::Py, args...; kwargs) = df.to_json(args...; kwargs...)
read_parquet(args...; kwargs...) = cudf.read_parquet(args...; kwargs...)
to_parquet(df::Py, args...; kwargs) = df.to_parquet(args...; kwargs...)
read_orc(args...; kwargs...) = cudf.read_orc(args...; kwargs...)
to_orc(df::Py, args...; kwargs) = df.to_orc(args...; kwargs...)
read_hdf(args...; kwargs...) = cudf.read_hdf(args...; kwargs...)
to_hdf(df::Py, args...; kwargs) = df.to_hdf(args...; kwargs...)
read_feather(args...; kwargs...) = cudf.read_feather(args...; kwargs...)
to_feather(df::Py, args...; kwargs) = df.to_feather(args...; kwargs...)
read_arvo(args...; kwargs...) = cudf.read_arvo(args...; kwargs...)


export
    # I/O
    read_csv,
    to_csv,
    read_text,
    read_json,
    to_json,
    read_parquet,
    to_parquet,
    read_orc,
    to_orc,
    read_hdf,
    to_hdf,
    read_feather,
    to_feather,
    read_arvo

end
