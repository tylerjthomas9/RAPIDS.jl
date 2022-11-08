
@testset "generic regression interface tests" begin
    X, y = MLJTestIntegration.make_regression()
    df_X = DataFrame(X)
    @testset "LinearRegression" begin
        failures, summary = MLJTestIntegration.test(
            [LinearRegression,],
            df_X, y;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=true, # set to true to debug
        )
        @test isempty(failures)
    end
end