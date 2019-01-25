# This function calculates the free kurtosis of a matrix
# The function is uniform for both Hermitian or rectangular X
function κ₄(X)
    #--------------------------------------------------------------------------
    # Syntax:       free_kurt = κ₄(X)
    # 
    # Input:        X: a matrix, Hermitian or rectangular.
    #
    # Outputs:      free_kurt: a scalar, the free kurtosis of X
    #   
    # Description:  This function return the free kurtosis of a matrix X
    # 
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------

    # get dimension
    N, M = size(X)
    
    # compute the free kurtosis
    return tr((X*X')^2)/N - (1 + N/M) * (tr(X*X')/N)^2
end

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

function free_ent(X; mat_type = "her")
    #--------------------------------------------------------------------------
    # Syntax:       chi = free_ent(X, "her")
    #               chi = free_ent(X, "rec")
    # 
    # Input:        X is a Hermitian or rectangular matrice
    #               mat_type is the type of X, valid options are "her" and "rec"
    #
    # Outputs:      chi is a scalr, it is the free entropy of X
    #   
    # Description:  This function return the free entropy of a given matrix X, 
    # 
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------

    # get dimension
    N, M = size(X)
    
    # compute the entropy
    # self-adjoint case
    if mat_type == "her"
        Λ = eigvals(Hermitian(X))
        ent = mean(log.(abs.([Λ[i] - Λ[j] for i = 1:N for j = (i + 1):N])))
    end
                            
    # rectangular case                       
    if mat_type == "rec"
        a, b = N/(N + M), M/(N + M)
        Λ = svdvals(X*X')  
        ent = a^2 * mean(log.(abs.([Λ[i] - Λ[j] for i = 1:N for j = (i + 1):N]))) + a*(b - a)*mean(log.(abs.(Λ)))
    end
                                                    
    return ent                                               
end
                                                    
function sum_free_ent(Z; mat_type = "her")
    #--------------------------------------------------------------------------
    # Syntax:       sum_chi = sum_free_ent(Z, "her")
    #               sum_chi = sum_free_ent(Z, "rec")
    # 
    # Input:        Z: a array of Hermitian or rectangular matrices
    #               mat_type is the type of X, valid options are "her" and "rec"
    #
    # Outputs:      sum_chi: a scalr, it is the summation of free entropy of Z[i]
    #   
    # Description:  This function return the summation of free entropy of all 
    #               components of the input
    # 
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------
    
    return sum(free_ent.(Z, mat_type = mat_type))                                                   
end