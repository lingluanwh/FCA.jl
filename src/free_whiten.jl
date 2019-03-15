"""
    mat_center(X; mat = "her")

This function centers input. 
If X is a Hermitian matrix, then ``Xcen = X - Tr(X)/N*I```.
If X is a rectangular matrix, then ``Xcen = X - X*ones(M,1)*ones(1,M)/M``.

# Arguments      
-   `X`: a Hermitian or rectangular matrix
-   `mat`: the type of `X`, valid options are "her" and "rec"

# Outputs:
-   `Xc`: centered version of `X`
"""
function mat_center(X::Array{T,2}; mat = "her") where T <: Number
    N, M = size(X)

    if mat == "her"
        # for Hermitian matrix, X = X - I*tr(X)/N
        if M != N
            error("X should be a symmetric matrix")
        end
        Xc = X - I*tr(X)/M
    elseif mat == "rec"
        # for rectangular matrix, A = A - mean(A)*ones(N, M)
        Xc = X - mean(X)*ones(N, M)
    end

    return Xc
end


"""
    free_whiten(Z; mat = "her")

This function whitens input array of matrices in the free probability sense

# Arguments
-   `Z`: an array of matrices of mat and of the same dimensions
-   `mat`: the type of Z[i], valid options are "her" and "rec"

# Outputs
-   `Y`: an array of centered matrices, whitened in the free probability sense, 
       that is, Tr(Y[i]*Y[j]')/size(Y[1],1) = (i == j)
-   `U`: a matrix of size s-by-s, where s = size(Z, 1)
-   `Σ`: a matrix of size s-by-s, 
"""
function free_whiten(Z::Array{Array{T, 2}}; mat = "her") where T <: Number
    Zc = mat_center.(Z; mat = mat)
    N = size(Zc[1], 1)

    # compute the cov matrix such that
    # cov[i,j] = tr(Zc[i]*Zc[j]')/N
    eigcov = eigen(real(tr.(Zc*Zc'/N)));
    Λ, U = eigcov.values, eigcov.vectors # only take the real part
    
    Λ[Λ.<0] .= 0 # keep non-negative eigs
    Σ = Diagonal(Λ).^0.5 
    Y = U*pinv(Σ)*U'*Zc

    return Y, U, Σ
end

# if Z is just a single matrix, we first convert it to [Y].
function free_whiten(Z::Array{T, 2}; mat = "her") where T <: Number
    return free_whiten([Z]; mat = mat)
end