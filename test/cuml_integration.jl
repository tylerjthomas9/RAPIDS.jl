
@testset "generic interface tests" begin
    @testset "Regression" begin
        failures, summary = MLJTestIntegration.test(
            [
                LinearRegression,
                Ridge,
                Lasso,
                ElasticNet,
                MBSGDRegressor,
                RandomForestRegressor,
                CD,
                # SVR,
                # LinearSVR,
                KNeighborsRegressor,
            ],
            MLJTestIntegration.make_regression()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false,
        )
        @test isempty(failures)
    end

    @testset "Classification" begin
        X, y = MLJTestIntegration.make_binary()
        failures, summary = MLJTestIntegration.test(
            [
                LogisticRegression,
                MBSGDClassifier,
                RandomForestClassifier,
                # SVC,
                # LinearSVC,
                KNeighborsClassifier,
            ],
            X,
            y;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=true,
        )
        @test isempty(failures)
    end
end
