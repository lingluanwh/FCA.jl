@testset "Test freecf $mat, $obj" for mat in ["her", "rec"], obj in ["kurt", "ent"]

    # test for the hermitian matrices
    if mat == "her"
        for idx = 1: 5
            # set up
            N = 300
            G1, G2 = randn(N, N), randn(N, 2N);
            X1, X2 = (G1 + G1') / sqrt(2*N), (G2 * G2') / (2*N)
            X1 = free_whiten([X1]; mat = "her")[1][1]
            X2 = free_whiten([X2]; mat = "her")[1][1]
            X = [X1, X2];
            A = randn(2,2)
            Z = A*X

            # apply freecf
            Aest = freecf(Z; mat = mat, obj = obj, opt = "orth")[1]
            @test err_pd(pinv(Aest)*A) < 5e-2
        end 
    end               

    # test for the rectangular matrices
    if mat == "rec"
        for idx = 1: 5
            # set up
            N, M = 300, 500
            X1 = randn(N, M) / sqrt(M)
            X1 = free_whiten([X1], mat = "rec")[1][1]
            U, V = Matrix(qr(randn(N,N)).Q), Matrix(qr(randn(M,M)).Q)
            D = [Diagonal((collect(range(1,stop = 0,length = N)) .- 1).^4) zeros(N, M - N)]
            X2 = U*D*V'
            X2 = free_whiten([X2], mat = "rec")[1][1]
            X = [X1, X2];
            A = randn(2,2)
            Z = A*X

            # apply freecf 
            Aest = freecf(Z; mat = mat, obj = obj, opt = "orth")[1]
            @test err_pd(pinv(Aest)*A) < 5e-2
        end
    end
end

@testset "Test freecf $mat, $obj" for mat in ["her", "rec"], obj in ["kurt"]

    # sphe opt is not designed for entropy based fcf

    # test for the hermitian matrices
    if mat == "her"
        for idx = 1: 5
            # set up
            N = 300
            G1, G2 = randn(N, N), randn(N, 2N);
            X1, X2 = (G1 + G1') / sqrt(2*N), (G2 * G2') / (2*N)
            X1 = free_whiten([X1]; mat = "her")[1][1]
            X2 = free_whiten([X2]; mat = "her")[1][1]
            X = [X1, X2];
            A = randn(2,2)
            Z = A*X

            # apply freecf
            Aest = freecf(Z; mat = mat, obj = obj, opt = "sphe")[1]
            @test err_pd(pinv(Aest)*A) < 5e-2
        end 
    end               

    # test for the rectangular matrices
    if mat == "rec"
        for idx = 1: 5
            # set up
            N, M = 300, 500
            X1 = randn(N, M) / sqrt(M)
            X1 = free_whiten([X1], mat = "rec")[1][1]
            U, V = Matrix(qr(randn(N,N)).Q), Matrix(qr(randn(M,M)).Q)
            D = [Diagonal((collect(range(1,stop = 0,length = N)) .- 1).^4) zeros(N, M - N)]
            X2 = U*D*V'
            X2 = free_whiten([X2], mat = "rec")[1][1]
            X = [X1, X2];
            A = randn(2,2)
            Z = A*X

            # apply freecf 
            Aest = freecf(Z; mat = mat, obj = obj, opt = "sphe")[1]
            @test err_pd(pinv(Aest)*A) < 5e-2
        end
    end
end