# test for mat_center function
@testset "Test mat_center function $mat_type" for mat_type in ["her", "rec"]
    # test for Hermitian matrix
    if mat_type == "her"
        for idx = 1: 10
            @test abs(tr(mat_center(randn(3,3); mat_type = mat_type))) <= 3*eps(Float64);
        end
    end

    # test for rectangular case
    if mat_type == "rec"
        for idx = 1: 10
            @test sum(abs.(mean(mat_center(randn(3,3); mat_type ="rec"), dims = 2))) <= 3*eps(Float64)
        end
    end
end

# test for free_whiten function
@testset "Test free_whiten function $mat_type" for mat_type in ["her", "rec"]
    # test for Hermitian matrix
    if mat_type == "her"
        for idx = 1: 10
            G = randn(10, 10)
            Z = [G + G', G*G']
            Y = free_whiten(Z; mat_type = "her")[1]
            @test maximum(abs.( tr.(Y*Y')./10 - I)) < 10*eps(Float64)
        end
    end

    # test for rectangular matrix
    if mat_type == "rec"
        for idx = 1: 10
            Z = [randn(10, 20) for i = 1: 3]
            Y = free_whiten(Z; mat_type = "rec")[1]
            @test maximum(abs.( tr.(Y*Y')./10 - I)) < 10*eps(Float64)
        end
    end
end