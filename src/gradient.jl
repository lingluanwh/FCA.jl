# function grad_neg_abs_sum_free_kurt(W::Array{T}, Y::Array{Array{T,2},1}) where T <: Number

"""
    grad_neg_abs_sum_free_kurt(W, Y)

This function calcualtes the gradient of neg_abs_sum_free_kurt(W'*Y) w.r.t W

# Arguments
-   `W`: a matrix such that size(`W`, 1) = size(`Y`, 1)
-   `Y`: an array of matrix of same type and same dimension

# Outputs
-   `grad`: the gradient of neg_abs_sum_free_kurt(W'*Y) w.r.t W
"""
function grad_neg_abs_sum_free_kurt(W, Y)
    # convert W to matrix 
    W = W[:, :]

    # get the number of components
    s = size(Y,1)
    
    # The first dimension of W should be s
    if size(W, 1) != s
        error("The dimensions do not match!")
    end
    
    # get the dimension of each matrix
    N, M = size(Y[1])
    
    # initialize grad matrix
    grad = complex(zeros(size(W)))
    
    # Compute X = W\otimes I * Y
    X = W'*Y
    
    # Compute the gradient
    for l = 1: size(grad, 2)
        sl = sign(κ₄(X[l]))
        for k = 1: size(grad, 1)
            grad[k, l] = -sl * (4 * tr(Y[k]*(X[l]'*X[l])*X[l]')/N 
                - 4 * (1 + N/M) * tr(X[l]*X[l]')/N * tr(Y[k]*X[l]')/N)
        end
    end
    return grad
end

# function grad_sum_free_ent(W::Array{T}, Y::Array{Array{T,2},1}, mat = "her") where T <: Number

"""
    grad_sum_free_ent(W, Z; mat = "her")

This function calcualtes the gradient of sum_free_ent(W'*Y) w.r.t W

# Arguments
-   `W`: a matrix such that size(W, 1) = size(Y, 1)
-   `Y`: an array of matrix of same type and same dimension

# Outputs
-   `grad`: the gradient of sum_free_ent(W'*Y) w.r.t W
"""
function grad_sum_free_ent(W, Z; mat = "her")
    # convert W to matrix
    W = W[:, :]
    
    # get the number of components
    s = size(Z,1)
    
    # The first dimension of W should be s
    if size(W, 1) != s
        error("The dimensions do not match!")
    end
    
    # get the dimension of each matrix
    N, M = size(Z[1])
    
    # initialize grad matrix
    grad = complex(zeros(size(W)))
    
    # Compute X = W\otimes I * Z
    X = W'*Z
    
    # Compute the gradient
    # self-adjoint case
    if mat == "her"
        for l = 1: size(grad, 2)
            eigres = eigen(X[l])
            Λ, V = eigres.values, eigres.vectors
            for k = 1: size(grad, 1)
                eiggrad = diag(V'*Z[k]*V)
                grad[k, l] = mean([(eiggrad[i] - eiggrad[j])/(Λ[i] - Λ[j]) for i = 1:N for j = (i+1):N])
            end
        end
    end
    
    # rectangular case                        
    if mat == "rec"
        a, b = N/(N + M), M/(N + M)
        for l = 1: size(grad, 2)
            eigres = eigen(X[l]*X[l]')
            Λ, V = eigres.values, eigres.vectors
            for k = 1: size(grad, 1)
                eiggrad = diag(V'*(Z[k]*X[l]' + X[l]*Z[k]')*V)
                grad[k, l] = (a^2 * mean([(eiggrad[i] - eiggrad[j])/(Λ[i] - Λ[j]) for i = 1:N for j = (i+1):N]) +
                              a*(b - a) * mean(eiggrad ./ Λ))  
            end
        end
    end
                                                                                                                
    return grad
end
