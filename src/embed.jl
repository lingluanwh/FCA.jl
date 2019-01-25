# introduce embedding between matrices (vectors) of different dimensions

function mat_embed(X, dim; zero_pos = -1, in_type = "her", out_type = "her")
    #--------------------------------------------------------------------------
    # Syntax:       Xnew = mat_embed(X, dim)
    #               chi = free_ent(X, dim)
    # 
    # Input:        X: a Hermitian or rectangular matrice
    #               dim: target dimension, array-like [Nnew,Mnew]
    #               zero_pos: an array of user specify Cartesian position for zeros. if 
    #               it is -1, then use randomly generated position.
    #               in_type: the type of input matrix, valid options are "her" and "rec".
    #                   if in_type is "her", only upper diagonal (include the diagonal) 
    #                   part of X is used
    #               out_type: the type of ouput matrix, valid options are "her" and "rec"
    #                   if out_type is "her", X is embedded to upper diagonal part of 
    #                   Xnew (exclude the diagonal).
    #
    # Outputs:      Xnew: a matrix of dimension dim and the type in_type. The entries 
    #               Xnew are from X, with possible extra zeros at zero_pos, the Cartesian
    #               order the entries are preserved.
    #               
    # Description:  This function embeds X into new dimensions dim. Fill in zeros at 
    #               zero_pos if necessary.  
    # 
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------

    # get the original dimension
    X = X[:,:]
    N, M = size(X)
    
    # get the target dimension
    Nnew, Mnew = dim
    
    # check the input dimension if mat_type is Hermitian
    if (in_type == "her") & (N != M)
        error("Input dimension is not for a Hermitian matrix!")
    end

    # reshape X into vector according to in_type
    if in_type == "her"
        in_size = N*(N + 1)/2
        Xvec = [X[i,j] for i=1:N for j=i:N]
    elseif in_type == "rec"
        in_size = N*M       
        Xvec = vec(X)       
    end

    # check whether there is enough room in the target shape
    if out_type == "her"
        out_size = Int((Nnew - 1)*Nnew/2)
    elseif out_type == "rec"
        out_size = Nnew*Mnew
    end

    if out_size < in_size
        error("The target dimension is too small!")
    end
    
    # if the position of zeros are not preset, generate one
    if zero_pos == -1
        zero_pos = sample(1:out_size, out_size - in_size, replace=false)
    end
                
    # Construct Xnewvec
    Xnewvec = zeros(out_size)
    Xnewvec[[i for i in 1: out_size if ~(i in zero_pos)]] = Xvec

    # reshape Xnewvec to Xnew
    if out_type == "her"
        k = 0                           
        Xnew = [i < j ? (k += 1; Xnewvec[k]) : 0 for i = 1: Nnew, j = 1: Mnew]
        Xnew = Xnew + Xnew'
    elseif out_type == "rec"
        Xnew = reshape(Xnewvec, (Nnew, Mnew))
    end
                                
    return convert(Array{typeof(X[1,1]), 2}, Xnew)
end