# Indepndent component factorization (ICF) based on Independent component analysis (ICA)
# We will compare FCF with ICF

include("orthopt.jl")

# Independent component factorization based on ICA
function icf(Z::Array{Array{D, 1}, 1}; opt_method = "orth") where D <: Number
    #--------------------------------------------------------------------------
    # Syntax:       Aest, Xest = icf(Z, opt_method = "orth")
    # 
    # Input:        Z: an array of vectors of the same length, represent the
    #                   realizations of mixed independent signals
    #               opt_method: string, valid option: "orth", "sphe"
    #                       
    # Outputs:      Aest: estimated mixing matrix of size s-by-s, where s = size(Z,1)
    #               Xest: estimated independent component Xest = pinv(Aest)*Z
    #   
    # Description:  Apply independent component analysis to Z, return estimated
    #               mixing matrix and independent components
    #
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------  

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

    # find the orthogonal matrix minimizer neg_abs_sum_kurt
    
    # optimization using optim
    if opt_method == "orth"
        # loss function
        F = W -> neg_abs_sum_kurt(W'*Y)

        # gradient of loss function
        grad_F = W -> grad_neg_abs_sum_kurt(W, Y)

        # using Stiefel manifold
        W = OptOrtho(F, grad_F, s)
    elseif opt_method == "sphe"
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
    
    Aest = U*Σ*U'*W
    xest = pinv(Aest) * Z
    return Aest, xest
end

function neg_abs_sum_kurt(Z)
    #--------------------------------------------------------------------------
    # Syntax:       F = neg_abs_sum_kurt(Z)
    # 
    # Input:        Z: an array of vectors of the same length, regarded as the
    #                   realizations of mixed independent signals
    #                       
    # Outputs:      F: a scalar, summation of negative absolute value 
    #                   of kurtosis of Z[i]
    #
    # Description:  Apply independent component analysis to Z, return estimated
    #               mixing matrix and independent components
    #
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #-------------------------------------------------------------------------- 

    return -sum(abs.(kurt.(Z)))
end

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

    return mean(z.^4) - 3*mean(z.^2)^2
end

function grad_neg_abs_sum_kurt(W, Z)
    #--------------------------------------------------------------------------
    # Syntax:       grad = grad_neg_abs_sum_kurt(W, Z)
    # 
    # Input:        W: a matrix such that size(W, 1) = size(Z, 1)
    #               Z: an array of vectors of the same length, regarded as the
    #                   realizations of mixed independent signals
    #                       
    # Outputs:      grad: the gradient of neg_abs_sum_kurt(W'*Z) w.r.t W
    #   
    # Description:  This function calcualtes the gradient of 
    #               neg_abs_sum_kurt(W'*Z) w.r.t W
    #
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------

    # get number of components and dimension 
    s = size(Z, 1)

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