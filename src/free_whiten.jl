function mat_center(X::Array{T,2}; mat_type = "her") where T <: Number
    #--------------------------------------------------------------------------
    # Syntax:       Xcen = mat_center(X, "her")
    #               Xcen = mat_center(X, "rec")
    # 
    # Input:        X: a Hermitian or rectangular matrix
    #               mat_type: the type of X, valid options are "her" and "rec"
    #
    # Outputs:      Xcen: a matrix of the same type as X. 
    #   
    # Description:  This function centers input. 
    #               If X is a Hermitian matrix, then Xcen = X - Tr(X)/N*I.
    #               If X is a rectangular matrix, then Xcen = X - X*ones(M,1)*ones(1,M)/M
    # 
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------

    N, M = size(X)
    if mat_type == "her"
        # for Hermitian matrix, X = X - I*tr(X)/N
        if M != N
            error("X should be a symmetric matrix")
        end
        return X - I*tr(X)/M
    elseif mat_type == "rec"
        # for rectangular matrix, A = A - A*ones(m, 1)*ones(1, m)/m
        return X - X*ones(M, 1)/M * ones(1, M)
    end
end


# for an array of matrix, whiten the array under the convention of free probability
function free_whiten(Z::Array{Array{T, 2}}; mat_type = "her") where T <: Number
    #--------------------------------------------------------------------------
    # Syntax:       Y, U, Σ = free_whiten(Z, "her")
    #               Y, U, Σ = free_whiten(Z, "rec")
    #
    # Input:        Z: an array of matrices of mat_type and of the same dimensions
    #               mat_type: the type of Z[i], valid options are "her" and "rec"
    #
    # Outputs:      Y: an array of centered matrices, whitened in the free probability sense,
    #                   that is, Tr(Y[i]*Y[j]')/size(Y[1],1) = (i == j)
    #               U: a matrix of size s-by-s, where s = size(Z, 1)
    #               Σ: a matrix of size s-by-s, 
    #   
    # Description:  This function whitens input array of matrices in the free probability sense
    # 
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------

    Zc = mat_center.(Z; mat_type = mat_type)
    N = size(Zc[1], 1)
    # compute the cov matrix such that
    # cov[i,j] = tr(Zc[i]*Zc[j]')/N
    eigcov = eigen(real(tr.(Zc*Zc'/N)));
    Λ, U = eigcov.values, eigcov.vectors # only take the real part
    
    Λ[Λ.<0] .= 0 # keep non-negative eigs
    Σ = Diagonal(Λ).^0.5 
    
    return U*pinv(Σ)*U'*Zc, U, Σ
end

# if Z is just a single matrix, we first convert it to [Y].
function free_whiten(Z::Array{T, 2}; mat_type = "her") where T <: Number
    return free_whiten([Z]; mat_type = mat_type)
end