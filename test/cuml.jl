const make_classification = cuml.datasets.classification.make_classification
const make_regression = cuml.datasets.regression.make_regression

# Clustering
@testset "cuML Clustering" begin
    X = rand(1000, 5)
    @testset "KMeans" begin
        model = KMeans()
        mach = machine(model, X)
        fit!(mach)
        preds = predict(mach, X)
    end

    @testset "DBSCAN" begin
        model = DBSCAN()
        mach = machine(model, X)
        fit!(mach)
        preds = mach.report.labels
    end

    @testset "AgglomerativeClustering" begin
        model = AgglomerativeClustering()
        mach = machine(model, X)
        fit!(mach)
        preds = mach.report.labels
    end

    @testset "HDBSCAN" begin
        model = HDBSCAN()
        mach = machine(model, X)
        fit!(mach)
        preds = mach.report.labels
    end
end

# Regression
@testset "cuML Regression" begin
    X_py, y_py = make_regression(; n_samples=200, n_features=5,
                                 n_informative=5, bias=0.0, noise=0.3)
    X = RAPIDS.pyconvert(Matrix{Float32}, X_py.get())
    y = RAPIDS.pyconvert(Vector{Float32}, y_py.get().flatten())

    @testset "LinearRegression" begin
        model = LinearRegression()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end

    @testset "Ridge" begin
        model = Ridge()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end

    @testset "Lasso" begin
        model = Lasso()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end

    @testset "ElasticNet" begin
        model = ElasticNet()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end

    @testset "MBSGDRegressor" begin
        model = MBSGDRegressor()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end

    @testset "RandomForestRegressor" begin
        model = RandomForestRegressor()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end

    @testset "CD" begin
        model = CD()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end

    @testset "SVR" begin
        model = SVR()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end

    #TODO: Figure out why LinearSVR fails
    # @testset "LinearSVR" begin
    #     model = LinearSVR()
    #     mach = machine(model, X, y)
    #     fit!(mach)
    #     preds = predict(mach, X)
    # end

    @testset "KNeighborsRegressor" begin
        model = KNeighborsRegressor()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end
end

# Classification
@testset "cuML Classification" begin
    X_py, y_py = make_classification(; n_samples=200, n_features=4,
                                     n_informative=2, n_classes=2)
    X = RAPIDS.pyconvert(Matrix{Float32}, X_py.get())
    y = RAPIDS.pyconvert(Vector{Float32}, y_py.get().flatten())

    @testset "LogisticRegression" begin
        model = LogisticRegression()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end

    @testset "MBSGDClassifier" begin
        model = MBSGDClassifier()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end

    @testset "RandomForestClassifier" begin
        model = RandomForestClassifier()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end

    #TODO: Figure out why SVC fails
    # @testset "SVC" begin
    #     model = SVC()
    #     mach = machine(model, X, y)
    #     fit!(mach)
    #     preds = predict(mach, X)
    # end

    #TODO: Figure out why LinearSVC fails
    # @testset "LinearSVC" begin
    #     model = LinearSVC()
    #     mach = machine(model, X, y)
    #     fit!(mach)
    #     preds = predict(mach, X)
    # end

    @testset "KNeighborsClassifier" begin
        model = KNeighborsClassifier()
        mach = machine(model, X, y)
        fit!(mach)
        preds = predict(mach, X)
    end
end

# @testset "cuML Dimensionality Reduction" begin
    # X = rand(1000, 5)

    # @testset "PCA" begin
    #     model = PCA()
    #     mach = machine(model, X)
    #     fit!(mach)
    #     X_trans = transform(mach, X)
    #     inverse_transform(mach, X)
    # end

    # @testset "IncrementalPCA" begin
    #     model = IncrementalPCA()
    #     mach = machine(model, X)
    #     fit!(mach)
    #     X_trans = transform(mach, X)
    # end

    # @testset "TruncatedSVD" begin
    #     model = TruncatedSVD(n_components=2)
    #     mach = machine(model, X)
    #     fit!(mach)
    #     X_trans = transform(mach, X)
    #     inverse_transform(mach, X)
    # end

    # @testset "UMAP" begin
    #     model = UMAP(n_components=2)
    #     mach = machine(model, X)
    #     fit!(mach)
    #     X_trans = transform(mach, X)
    # end

    # @testset "GaussianRandomProjection" begin
    #     model = GaussianRandomProjection(n_components=2)
    #     mach = machine(model, X)
    #     fit!(mach)
    #     X_trans = transform(mach, X)
    # end

    # @testset "TSNE" begin
    #     model = TSNE(n_components=2)
    #     mach = machine(model, X)
    #     fit!(mach)
    #     X_trans = transform(mach, X)
    # end
# end

@testset "Time Series" begin
    X = [1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12,
    2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13,
    3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14] 
    @testset "ExponentialSmoothing" begin
        model = ExponentialSmoothing()
        mach = machine(model, X)
        fit!(mach)
        forecast(mach, 4)
    end
    @testset "ARIMA" begin
        model = ARIMA()
        mach = machine(model, X)
        fit!(mach)
        forecast(mach, 4)
    end
end


