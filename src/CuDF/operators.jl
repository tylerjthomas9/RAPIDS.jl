import Base: ==, >, <, >=, <=, !=

for (op, pyop) in
    [(:(==), :__eq__), (:>, :__gt__), (:<, :__lt__), (:>=, :__ge__), (:<=, :__le__),
     (:!=, :__ne__)]
    @eval function Base.broadcast(::typeof($op), s::PandasWrapped, x)
        method = s.pyo.$(QuoteNode(pyop))
        return pandas_wrap(pycall(method, x))
    end
end
