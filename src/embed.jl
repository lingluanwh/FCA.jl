# introduce embedding between matrices (vectors) of different dimensions

"""
    mat_embed(X, dim; zero_pos = -1, tpin = "her", tpout = "her")

This function embeds X into new dimensions dim, with the Cartesian
order of the entries are preserved. Fill in zeros at zero_pos if necessary.  

# Arguments       
-   `X`: a Hermitian or rectangular matrice              
-   `dim`: target dimension, array-like [Nnew,Mnew]
-   `zero_pos`: an array of user specify Cartesian position for zeros. if 
               it is -1, then use randomly generated position.
-   `tpin`: the type of input matrix, valid options are "her" and "rec".                        
-   `tpout`: the type of ouput matrix, valid options are "her" and "rec"
      
# Outputs     
-   `Xnew`: a matrix of dimension dim and the type tpin. The entries 
          Xnew are from X, with possible extra zeros at zero_pos,               
"""
function mat_embed(X, dim; zero_pos = -1, tpin = "her", tpout = "her")

    # get the original dimension
    X = X[:,:]
    N, M = size(X)
    
    # get the target dimension
    Nnew, Mnew = dim
    
    # check the input dimension if mat is Hermitian
    if (tpin == "her") & (N != M)
        error("Input dimension is not for a Hermitian matrix!")
    end

    # reshape X into vector according to type in
    if tpin == "her"
        in_size = N*(N + 1)/2
        Xvec = [X[i,j] for i=1:N for j=i:N]
    elseif tpin == "rec"
        in_size = N*M       
        Xvec = vec(X)       
    end

    # check whether there is enough room in the target shape
    if tpout == "her"
        out_size = Int((Nnew - 1)*Nnew/2)
    elseif tpout == "rec"
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
    if tpout == "her"
        k = 0                           
        Xnew = [i < j ? (k += 1; Xnewvec[k]) : 0 for i = 1: Nnew, j = 1: Mnew]
        Xnew = Xnew + Xnew'
    elseif tpout == "rec"
        Xnew = reshape(Xnewvec, (Nnew, Mnew))
    end
                                
    return convert(Array{typeof(X[1,1]), 2}, Xnew)
end