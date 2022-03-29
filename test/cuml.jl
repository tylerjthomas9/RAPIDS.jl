
# Clustering
x = rand(1000, 5)
@testset "KMeans" begin
    model = KMeans()
    mach = machine(model, x)
    fit!(mach)
    preds = predict(mach, x)
    @test typeof(preds) == Vector{Int}
end

@testset "DBSCAN" begin
    model = DBSCAN()
    mach = machine(model, x)
    fit!(mach)
end

@testset "AgglomerativeClustering" begin
    model = AgglomerativeClustering()
    mach = machine(model, x)
    fit!(mach)
end


@testset "HDBSCAN" begin
    model = HDBSCAN()
    mach = machine(model, x)
    fit!(mach)
end

# Regression
x = rand(1000, 5)
y = rand(1000)
@testset "LinearRegression" begin
    model = LinearRegression()
    mach = machine(model, x, y)
    fit!(mach)
    preds = predict(mach, x)
    @test typeof(preds) == Vector{Float32}
end

@testset "Ridge" begin
    model = Ridge()
    mach = machine(model, x, y)
    fit!(mach)
    preds = predict(mach, x)
    @test typeof(preds) == Vector{Float32}
end

@testset "Lasso" begin
    model = Lasso()
    mach = machine(model, x, y)
    fit!(mach)
    preds = predict(mach, x)
    @test typeof(preds) == Vector{Float32}
end


@testset "ElasticNet" begin
    model = ElasticNet()
    mach = machine(model, x, y)
    fit!(mach)
    preds = predict(mach, x)
    @test typeof(preds) == Vector{Float32}
end

@testset "MBSGDRegressor" begin
    model = MBSGDRegressor()
    mach = machine(model, x, y)
    fit!(mach)
    preds = predict(mach, x)
    @test typeof(preds) == Vector{Float32}
end

@testset "RandomForestRegressor" begin
    model = RandomForestRegressor()
    mach = machine(model, x, y)
    fit!(mach)
    preds = predict(mach, x)
    @test typeof(preds) == Vector{Float32}
end

@testset "CD" begin
    model = CD()
    mach = machine(model, x, y)
    fit!(mach)
    preds = predict(mach, x)
    @test typeof(preds) == Vector{Float32}
end

@testset "SVR" begin
    model = SVR()
    mach = machine(model, x, y)
    fit!(mach)
    preds = predict(mach, x)
    @test typeof(preds) == Vector{Float32}
end

@testset "LinearSVR" begin
    model = LinearSVR()
    mach = machine(model, x, y)
    fit!(mach)
    preds = predict(mach, x)
    @test typeof(preds) == Vector{Float32}
end

@testset "KNeighborsRegressor" begin
    model = KNeighborsRegressor()
    mach = machine(model, x, y)
    fit!(mach)
    preds = predict(mach, x)
    @test typeof(preds) == Vector{Float32}
end



# Classification