function freecf(Z; mat_type="her", obj_type="kurt", opt_method="orth")
    #--------------------------------------------------------------------------
    # Syntax:       Aest, Xest = freecf(Z, obj_type = "ent")
    #               Aest, Xest = freecf(Z, mat_type = "rec")
    #               Aest, Xest = freecf(Z, opt_method = "sphe")
    #
    # Input:        Z: an array of matrices of mat_type and of the same dimensions
    #               mat_type: the type of Z[i], valid options are "her" and "rec"
    #               obj_type: the type of loss function, valid options are "kurt" and "ent"
    #               opt_method: string, valid option: "orth", "sphe"
    #
    # Outputs:      Aest: a matrix of size s-by-s
    #               Xest: an array of "free" matrices, such that Z = Aest*Xest
    #   
    # Description:  Apply free component analysis to Z, return estimated
    #               mixing matrix and free components
    # 
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------
    
    # freecf is not designed for obj_type = "ent" and opt_method = "sphe"
    if (obj_type == "ent") & (opt_method == "sphe")
        error("Spherical optimization is not designed for entropy based fcf!")
    end

    # get the dimension of each matrix
    N, M = size(Z[1])
    s = size(Z, 1)
    
    # check whether the dimension with the mat_type
    if (mat_type == "sym") && (N != M)
        error("The input components are not a Hermitian matrix")
    end
    
    # whiten the data
    Y, U, Σ = free_whiten(Z; mat_type = mat_type);
    
    # assign the objective function and derivative according to 
    # obj_type and mat_type
    if obj_type == "kurt"
        # if we use kurtosis based cost function
        # obj function
        F = (W, Y) -> neg_abs_sum_free_kurt(W[:,:]'*Y);
        
        # derivative of obj function
        grad_F = (W, Y) -> grad_neg_abs_sum_free_kurt(W, Y);
        
    elseif obj_type == "ent"
        # if we use entropy based cost function
        # obj function
        F = (W, Y) -> sum_free_ent(W[:,:]'*Y; mat_type = mat_type);
        
        # derivative of obj function
        grad_F = (W, Y) -> grad_sum_free_ent(W, Y; mat_type = mat_type);
    end
    
    # optimization over the orthogonal matrix
    if opt_method == "orth"
        What = OptOrtho(W -> F(W, Y), W -> grad_F(W, Y), s);
    elseif opt_method == "sphe"
        # recover the columns of W one-by-one 
        # first column
        What = reshape(OptSphere(W -> F(W, Y), W -> grad_F(W, Y), s), :, 1)

        # find 2nd to s-1th column one at a time
        for i = 2: (s - 1)
            pY = (I - What*What')*Y

            # updated loss function
            
            # find i th column, attach it to W
            Wi = OptSphere(W -> F(W, pY), W -> grad_F(W, pY), s)
            What = (qr(hcat(What, Wi)).Q)[:, 1:i]
        end

        # find the s-th column by take qr decomposition
        What = Matrix(qr(hcat(What, randn(s))).Q)
    end
    
    # compute and return the estimated mixing matrix 
    # and free components
    Aest = U*Σ*U'*What;
    Xest = pinv(Aest) * Z;
    return Aest, Xest
end