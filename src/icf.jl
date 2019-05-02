# Indepndent component factorization (ICF) based on Independent component analysis (ICA)
# We will compare FCF with ICF

"""
    icf(Z; opt = "orth")

Apply independent component analysis to Z, return estimated 
mixing matrix and independent components

# Arguments
-   `Z`: an array of vectors of the same length, represent the
       realizations of mixed independent signals
-   `obj`: the type of loss function, valid options are "kurt" and "ent"
-   `opt`: string, type of optimization, valid option: "orth", "sphe". 
        The "sphe" is only designed for "kurt" loss function

# Outputs
-   `Aest`: estimated mixing matrix of size `s`-by-`s`, where `s` = size(Z,1)
-   `Xest`: estimated independent component `Xest = pinv(Aest)*Z`
"""
function icf(Z::Array{Array{D, 1}, 1}; obj = "kurt", opt = "orth") where D <: Number
    # get number of the components and the dimension of each components
    s = size(Z,1)
    T = size(Z[1],1)

    # center every components
    Zbar = mean.(Z)
    Zc = Z .- [Zbar[i]*ones(T) for i in 1:s]

    # compute the covariance matrix
    # whiten the Zc
    C = [Zc[i]'*Zc[j]/T for i = 1:s, j = 1:s]
    eigcov = eigen(C);
    Λ, U = eigcov.values, eigcov.vectors
    Λ[Λ.<0] .= 0 # keep non-negative eigs
    Σ = Diagonal(Λ).^0.5 
    Y = U*pinv(Σ)*U'*Zc
    
    if obj == "kurt"
        # find the orthogonal matrix minimizer neg_abs_sum_kurt

        # optimization using optim
        if opt == "orth"
            # loss function
            F = W -> neg_abs_sum_kurt(W'*Y)

            # gradient of loss function
            grad_F = W -> grad_neg_abs_sum_kurt(W, Y)

            # using Stiefel manifold
            W = OptOrtho(F, grad_F, s)
        elseif opt == "sphe"
            # using Sphere manifold and find one column at a time
            # loss function
            F = W -> neg_abs_sum_kurt(W[:,:]'*Y)

            # gradient of loss function
            grad_F = W -> grad_neg_abs_sum_kurt(W, Y)

            # first column
            W = reshape(OptSphere(F, grad_F, s), :, 1)

            # find 2nd to s-1th column one at a time
            for i = 2: s - 1
                pY = (I - W*W')*Y

                # updated loss function
                F = W -> neg_abs_sum_kurt(W[:,:]'*pY)

                # updated gradient of loss function
                grad_F = W -> grad_neg_abs_sum_kurt(W, pY)

                # find i th column, attach it to W
                Wi = OptSphere(F, grad_F, s)
                W = (qr(hcat(W, Wi)).Q)[:, 1:i]
            end

            # find the s-th column by take qr decomposition
            W = Matrix(qr(hcat(W, randn(s))).Q)
        end
    elseif obj == "ent"
        F = W -> ent_sum(W'*Y)
        
        grad_F = W -> grad_ent_sum(W, Y)
        
        # using Stiefel manifold
        W = OptOrtho(F, grad_F, s)
    end
    
    Aest = U*Σ*U'*W
    xest = pinv(Aest) * Z

    # order Xest by the descending order of absolute free kurtosis if obj = "kurt"
    if obj == "kurt"
        order = sortperm(xest, by = x -> -abs(kurt(x)))
    elseif obj == "ent"
        order = sortperm(xest, by = x -> -negent(x))
    end
    
    xest = xest[order]
    Aest = Aest[:,order]

    return Aest, xest
end


"""
    neg_abs_sum_kurt(Z)

Calculate the sum of negative absolute value of kurtosis of rows of Z
"""
function neg_abs_sum_kurt(Z)
    return -sum(abs.(kurt.(Z)))
end

"""
    kurt(z)

This function returns the kurtosis of z   

# Arguments
-   `z`: an vector regarded as independent realizations of a random variables

# Outputs
-   `k`: a scalar, kurtosis of z
"""
function kurt(z)
    #--------------------------------------------------------------------------
    # Syntax:       k = kurt(z)
    # 
    # Input:        z: an vector regarded as independent realizations of a random variables
    #                       
    # Outputs:      k: a scalar, kurtosis of z
    #
    # Description:  This function returns the kurtosis of z
    #
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #-------------------------------------------------------------------------- 
    z = z .- mean(z)
    k = mean(z.^4) - 3*mean(z.^2)^2
    return k
end

"""
grad_neg_abs_sum_kurt(W, Z)

This function calcualtes the gradient of neg_abs_sum_kurt(W'*Z) w.r.t W

# Arguments       
-   `W`: a matrix such that size(W, 1) = size(Z, 1)
-   `Z`: an array of vectors of the same length, regarded as the
       realizations of mixed independent signals

# Outputs
-   `grad`: the gradient of neg_abs_sum_kurt(W'*Z) w.r.t W
"""
function grad_neg_abs_sum_kurt(W, Z)
    # get number of components and dimension 
    s = size(Z, 1)

    # center rows of Z
    Z = [Z[i] .- mean(Z[i]) for i = 1: s]
    
    # initalize grad matrix
    grad = complex(zeros(size(W)))

    # Compute the gradient
    X = (W[:,:])'*Z

    for l = 1: size(grad , 2)
        sl = sign(kurt(X[l]))
        for k = 1: size(grad, 1)
            grad[k, l] = -sl * (4 * mean(Z[k].*(X[l].^3)) - 12 * mean(X[l].^2)*mean(Z[k].*X[l]))
        end
    end

    return grad
end

"""
negent(z)

This function returns the negentropy of z   

# Arguments
-   `z`: an vector regarded as independent realizations of a random variables

# Outputs
-   `k`: a scalar, negentropy of z
"""
function negent(z)
    # approximate the entropy of z by cumulants
    z = z .- mean(z); # center z
    return kurt(z)^2/48 + mean(z.^3)^2/12
end

"""
    ent_sum(Z)

Calculate the negative um of negentropy of rows of Z
"""
function ent_sum(Z)
    return -sum(negent.(Z))
end

"""
grad_ent_sum(W, Z)

This function calcualtes the gradient of ent_sum(W'*Z) w.r.t W

# Arguments       
-   `W`: a matrix such that size(W, 1) = size(Z, 1)
-   `Z`: an array of vectors of the same length, regarded as the
       realizations of mixed independent signals

# Outputs
-   `grad`: the gradient of ent_sum(W'*Z) w.r.t W
"""
function grad_ent_sum(W, Z)
    # get number of components and dimension 
    s = size(Z, 1)

    # center rows of Z
    Z = [Z[i] .- mean(Z[i]) for i = 1: s]

    # initalize grad matrix
    grad = complex(zeros(size(W)))

    # Compute the gradient
    X = (W[:,:])'*Z

    for l = 1: size(grad , 2)
        for k = 1: size(grad, 1)
            grad[k, l] =  -1 * kurt(X[l])/24 * (4 * mean(Z[k].*(X[l].^3)) - 12 * mean(X[l].^2)*mean(Z[k].*X[l])) 
            grad[k, l] -= mean(X[l].^3)/6 * (3 * mean(Z[k] .* (X[l].^2)))
        end
    end

    return grad
end