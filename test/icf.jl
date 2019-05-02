# test function kurt
x = [-1,0,2,-1]
@test kurt(x) == 18/4 - 3*(6/4)^2

x = [0]
@test kurt(x) == 0

x = [2,1,3,-2,-1,-3]
@test kurt(x) == mean(x.^4) - 3*mean(x.^2)^2

# test function grad_neg_sum_kurt()
@testset "Test grad_neg_sum_kurt" begin
    for idx = 1: 10
        Z = [randn(100000) for i = 1: 3]
        W = randn(3,3)
        @test rel_error(ForwardDiff.gradient(x -> neg_abs_sum_kurt(x'*Z), W), grad_neg_abs_sum_kurt(W, Z)) < 1e-8
    end
end

@testset "Test kurtosis-based icf $opt" for opt in ["orth", "sphe"]  
    for idx = 1: 10
        # set up
        T = 100000
        x = [rand(T) + 1*ones(T), rand(T) - 1*ones(T)]
        A = randn(2,2)
        z = A*x;

        # apply icf
        Aest = icf(z; opt = opt)[1]
        @test err_pd(pinv(Aest)*A) < 5e-2
    end
end

# test function grad_ent_sum()
@testset "Test grad_ent_sum" begin
    for idx = 1: 10
        Z = [randn(100000) for i = 1: 3]
        W = randn(3,3)
        @test rel_error(ForwardDiff.gradient(x -> ent_sum(x'*Z), W), grad_ent_sum(W, Z)) < 1e-8
    end
end

@testset "Test entropy-based icf" begin
    for idx = 1: 10
        # set up
        T = 100000
        x = [rand(T) + 1*ones(T), rand(T) - 1*ones(T)]
        A = randn(2,2)
        z = A*x;

        # apply icf
        Aest = icf(z; obj = "ent")[1]
        @test err_pd(pinv(Aest)*A) < 5e-2
    end
end
