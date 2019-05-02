# test for mat_center function
@testset "Test mat_center function $mat" for mat in ["her", "rec"]
    # test for Hermitian matrix
    if mat == "her"
        for idx = 1: 10
            @test abs(tr(mat_center(randn(3,3); mat = mat))) <= 10*eps(Float64);
        end
    end

    # test for rectangular case
    if mat == "rec"
        for idx = 1: 10
            @test sum(vec(mat_center(randn(3,3); mat ="rec"))) <= 10*eps(Float64)
        end
    end
end

# test for free_whiten function
@testset "Test free_whiten function $mat" for mat in ["her", "rec"]
    # test for Hermitian matrix
    if mat == "her"
        for idx = 1: 10
            G = randn(10, 10)
            Z = [G + G', G*G']
            Y = free_whiten(Z; mat = "her")[1]
            @test maximum(abs.( tr.(Y*Y')./10 - I)) < 10*eps(Float64)
        end
    end

    # test for rectangular matrix
    if mat == "rec"
        for idx = 1: 10
            Z = [randn(10, 20) for i = 1: 3]
            Y = free_whiten(Z; mat = "rec")[1]
            @test maximum(abs.( tr.(Y*Y')./10 - I)) < 10*eps(Float64)
        end
    end
end