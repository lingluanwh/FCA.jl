push!(LOAD_PATH, "../src/")
using FCA
using Test
using Combinatorics, ForwardDiff, LinearAlgebra, Statistics

# function calculates relative error
function rel_error(x, y)
    return findmax(abs.(x - y) ./ (max.(1e-8, abs.(x) + abs.(y))))[1]
end

tests = [
    "embed.jl",
    "free_whiten.jl",
    "gradient.jl",
    "icf.jl",
    "fcf.jl"
]

# test embed
@testset "Test $script" for script in tests
    println(script)
    @time include(script)
end