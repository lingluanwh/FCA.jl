# This function calculates the free kurtosis of a matrix
# The function is uniform for both Hermitian or rectangular X
"""
    κ₄(X)

This function return the free kurtosis of a matrix X

# Arguments
-   `X`: a matrix, Hermitian or rectangular

# Outputs
-   `fk`: a scalar, the free kurtosis of X
"""
function κ₄(X)
    # get dimension
    N, M = size(X)
    
    # compute the free kurtosis
    fk = tr((X*X')^2)/N - (1 + N/M) * (tr(X*X')/N)^2
    return fk
end

"""
    neg_abs_sum_free_kurt(Z)

This function calculates the sum of negative absolute value of free kurtosis of components of Z
"""
function neg_abs_sum_free_kurt(Z)
    #--------------------------------------------------------------------------
    # Syntax:       F = neg_abs_sum_free_kurt(Z)
    # 
    # Input:        Z: a array of Hermitian or rectangular matrices
    #
    # Outputs:      F: a scalr, it is the summation of negative absolute values 
    #               of the free kurtosis of Z[i]
    #   
    # Description:  This function return the summation of negative absolute value 
    #               of free kurtosis of input array of matrices
    # 
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------

    return -sum(abs.(κ₄.(Z)))
end

"""
    free_ent(X; mat = "her")

This function return the free entropy of a matrix X

# Arguments
-   `X`: a matrix, Hermitian or rectangular
-   `mat`: the type of X, valid options are "her" and "rec"

# Outputs
-   `chi`: a scalr, it is the free entropy of X
"""
function free_ent(X; mat = "her")
    # get dimension
    N, M = size(X)
    
    # compute the entropy
    # self-adjoint case
    if mat == "her"
        Λ = eigvals(Hermitian(X))
        chi = mean(log.(abs.([Λ[i] - Λ[j] for i = 1:N for j = (i + 1):N])))
    end
                            
    # rectangular case                       
    if mat == "rec"
        a, b = N/(N + M), M/(N + M)
        Λ = svdvals(X*X')  
        chi = a^2 * mean(log.(abs.([Λ[i] - Λ[j] for i = 1:N for j = (i + 1):N]))) + a*(b - a)*mean(log.(abs.(Λ)))
    end
                                                    
    return chi                                              
end

"""
    sum_free_ent(Z; mat = "her")

This function calculates the sum of free entropy of components of Z
"""
function sum_free_ent(Z; mat = "her")  
    return sum(free_ent.(Z, mat = mat))                                                   
end