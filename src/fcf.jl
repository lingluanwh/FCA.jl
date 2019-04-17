"""      
    freecf(Z; mat="her", obj="kurt", opt="orth")

Apply free component analysis to Z, return estimated
mixing matrix and free components

# Arguments      
-   `Z`: an array of matrices of mat and of the same dimensions
-   `mat`: the type of Z[i], valid options are "her" and "rec"
-   `obj`: the type of loss function, valid options are "kurt" and "ent"
-   `opt`: string, type of optimization, valid option: "orth", "sphe"

# Outputs     
-   `Aest`: a matrix of size `s`-by-`s`, where `s` = size(Z, 1)
-   `Xest`: an array of "free" matrices, such that `Z = Aest*Xest`
"""
function freecf(Z; mat="her", obj="kurt", opt="orth")
    
    # freecf is not designed for obj = "ent" and opt = "sphe"
    if (obj == "ent") & (opt == "sphe")
        error("Spherical optimization is not designed for entropy based fcf!")
    end

    # get the dimension of each matrix
    N, M = size(Z[1])
    s = size(Z, 1)
    
    # check whether the dimension with the mat
    if (mat == "her") && (N != M)
        error("The input components are not a Hermitian matrix")
    end
    
    # whiten the data
    Y, U, Σ = free_whiten(Z; mat = mat);
    
    # assign the objective function and derivative according to 
    # obj and mat
    if obj == "kurt"
        # if we use kurtosis based cost function
        # obj function
        F = (W, Y) -> neg_abs_sum_free_kurt(W[:,:]'*Y);
        
        # derivative of obj function
        grad_F = (W, Y) -> grad_neg_abs_sum_free_kurt(W, Y);
        
    elseif obj == "ent"
        # if we use entropy based cost function
        # obj function
        F = (W, Y) -> sum_free_ent(W[:,:]'*Y; mat = mat);
        
        # derivative of obj function
        grad_F = (W, Y) -> grad_sum_free_ent(W, Y; mat = mat);
    end
    
    # optimization over the orthogonal matrix
    if opt == "orth"
        What = OptOrtho(W -> F(W, Y), W -> grad_F(W, Y), s);
    elseif opt == "sphe"
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

    # order Xest by the descending order of absolute free kurtosis if obj = "kurt"
    if obj == "kurt"
        order = sortperm(Xest, by = X -> -abs(κ₄(X)))
        Xest = Xest[order]
        Aest = Aest[:,order]
    # order Xest by the aescending order of the free entropy
    elseif obj == "ent"
        order = sortperm(Xest, by = X -> free_ent(X; mat=mat))
        Xest = Xest[order]
        Aest = Aest[:,order]
    end

    return Aest, Xest
end
