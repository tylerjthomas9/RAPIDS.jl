
@testset "generic interface tests" begin
    @testset "Regression" begin
        failures, summary = MLJTestInterface.test([LinearRegression,
                                                   Ridge,
                                                   Lasso,
                                                   ElasticNet,
                                                   MBSGDRegressor,
                                                   RandomForestRegressor,
                                                   CD,
                                                   # SVR,
                                                   # LinearSVR,
                                                   KNeighborsRegressor],
                                                  MLJTestInterface.make_regression()...;
                                                  mod=@__MODULE__,
                                                  verbosity=1, # bump to debug
                                                  throw=false)
        @test isempty(failures)
    end

    @testset "Binary Classification" begin
        X, y_string = MLJTestInterface.make_binary()
        # RAPIDS can only handle numeric values
        # TODO: add support for non-numeric labels
        y = zeros(200)
        y[y_string .== "O"] .= 1.0
        failures, summary = MLJTestInterface.test([
                                                   #LogisticRegression,
                                                   MBSGDClassifier,
                                                   RandomForestClassifier,
                                                   #SVC,
                                                   #LinearSVC,
                                                   KNeighborsClassifier],
                                                  X,
                                                  y;
                                                  mod=@__MODULE__,
                                                  verbosity=1, # bump to debug
                                                  throw=false)
        @test isempty(failures)
    end

    @testset "Multiclass Classification" begin
        X, y_string = MLJTestInterface.make_multiclass()
        # RAPIDS can only handle numeric values
        # TODO: add support for non-numeric labels
        y = zeros(150)
        y[y_string .== "versicolor"] .= 1.0
        y[y_string .== "virginica"] .= 2.0
        failures, summary = MLJTestInterface.test([LogisticRegression,
                                                   RandomForestClassifier,
                                                   KNeighborsClassifier],
                                                  X,
                                                  y;
                                                  mod=@__MODULE__,
                                                  verbosity=0, # bump to debug
                                                  throw=false)
        @test isempty(failures)
    end
end
