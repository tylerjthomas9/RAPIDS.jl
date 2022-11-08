
@testset "generic interface tests" begin
    @testset "Regression" begin
        failures, summary = MLJTestIntegration.test(
            [LinearRegression,
            Ridge,
            Lasso,
            ElasticNet,
            MBSGDRegressor,
            RandomForestRegressor,
            CD, 
            # SVR,
            # LinearSVR,
            KNeighborsRegressor],
            MLJTestIntegration.make_regression()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false, # set to true to debug
        )
        @test isempty(failures)
    end
end