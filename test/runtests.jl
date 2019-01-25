push!(LOAD_PATH, "../src/")
using fca
using Test
using Combinatorics, ForwardDiff, LinearAlgebra, Statistics

# function calculates relative error
function rel_error(x, y)
    return findmax(abs.(x - y) ./ (max.(1e-8, abs.(x) + abs.(y))))[1]
end

# function calculate how close a matrix is to identity (up to permutations and rescaling) 
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

    return minimum(err)/sqrt(s)
end

tests = [
    "fcf.jl",
    "embed.jl",
    "free_whiten.jl",
    "gradient.jl",
    "icf.jl",
]

# test embed
@testset "Test $script" for script in tests
    println(script)
    @time include(script)
end