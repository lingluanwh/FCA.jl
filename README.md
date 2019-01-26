# fca.jl

fca.jl is a package for [julia](https://julialang.org/) implementing [free component analysis](https://ieeexplore.ieee.org/document/7868999), which separates freely-independent random matrices out of their additive mixture.
## Installation
The package has not been registered in `METADATA.jl` and can be installed with `Pkg.clone`.
```julia
julia> Pkg.clone("https://github.com/lingluanwh/fca.jl.git")
```
## Example
A typical example of the usage of fca.jl is
```julia
# separate symetric freely independent random matrices out of their additive mixture
using fca

# generate freely two symmetric freely independent random matrices X1 and X2
N = 300
G1, G2 = randn(N,N), randn(N,2*N)
X1 = (G1+G1')/sqrt(N)
X2 = (G2*G2')/2N

# mix X1, X2 linearly
A = randn(2,2) # mixing matrix
X = [X1, X2]
Z = A*X

# recover mixing matrix and free components using freecf
Aest, Xest = freecf(Z)

# Aest recover A upto column permutation and column rescaling.
using LinearAlgebra
@show pinv(Aest)*A # their product should approximate a diagonal matrix
```
