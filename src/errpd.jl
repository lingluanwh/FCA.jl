# function calculate how close a matrix is to identity (up to permutations and rescaling) 
using Combinatorics, LinearAlgebra
"""
    err_pd(Ie)

Calculate the error (up to permuataion and rescaling) between Ie and identity matrix:

```math
err = \\mathrm{min}_{P, D} ||PDI_{e} - I||_{F},
```

where ``P`` runs over all permuataion matrices and `` D `` runs over all diagonal non-singular matrices.
``||~||_F`` denotes the Frobeneous norm. 

# Arguments
-   `Ie`: the input matrix, supposed to be close to identity matrix.

# Outputs
-   `err`: the error between `Ie` and identity matrix defined as above.
"""
function err_pd(Ie)
    # get dimension
    s = size(Ie, 1)

    # generate all permutations in Sym(s)
    perm = permutations(1:s) |> collect

    # identity matrix
    idm = Matrix{Float64}(I, s, s)

    err = []
    for i in 1:factorial(s)
        Iep = Ie[perm[i],:]
        errtemp = []
        for j in 1: s
            eie = Iep[:,j]
            ei = idm[:,j]
            push!(errtemp, norm((I - eie*eie'/norm(eie)^2)*ei))
        end
        push!(err, norm(errtemp))
    end

    return minimum(err)
end