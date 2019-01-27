# require script: loss.jl, gradient.jl
# require package: ForwardDiff, LinearAlgebra, GenericLinearAlgebra, Statistics

# test test_grad_kurt function
@testset "Test test_grad_kurt" begin
    for idx = 1: 10
        s = 3
        Y = [randn(10,20) for i = 1: s]
        W = randn(s, s)
        @test rel_error(ForwardDiff.gradient(x -> neg_abs_sum_free_kurt(x'*Y), W)
            , grad_neg_abs_sum_free_kurt(W, Y)) < 1e-8
    end
end

# test grad_sum_free_ent function
@testset "Test grad_sum_free_ent $mat" for mat in ["her", "rec"]
    # test for Hermitian matrices
    if mat == "her"
        G = randn(10,10)
        Y = [G + G', G*G']
        W = randn(2,2)
        @test rel_error(ForwardDiff.gradient(x -> sum_free_ent(x'*Y; mat = "her"), W)
        , grad_sum_free_ent(W, Y; mat = "her")) < 1e-8
    end

    # test for rectangular matrices
    if mat == "rec"
        s = 3
        Y = [randn(10,20) for i = 1: s]
        W = randn(s,s)
        @test rel_error(ForwardDiff.gradient(x -> sum_free_ent(x'*Y; mat = "rec"), W)
            , grad_sum_free_ent(W, Y; mat = "rec")) < 1e-8
    end
end