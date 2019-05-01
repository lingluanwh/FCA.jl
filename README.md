# FCA.jl
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2655935.svg)](https://doi.org/10.5281/zenodo.2655935)

FCA.jl is a package for [julia](https://julialang.org/) implementing [free component analysis](https://ieeexplore.ieee.org/document/7868999), which separates freely-independent random matrices out of their additive mixture.
## Installation
The package has not been registered in `METADATA.jl` and can be installed with `Pkg.clone`.
```julia
julia> Pkg.clone("https://github.com/lingluanwh/FCA.jl")
```
## Example
A typical example of the usage of fca.jl is
```julia
# separate freely independent rectangular random matrices out of their additive mixture
using FCA, LinearAlgebra

# generate freely two freely independent rectangular random matrices X1 and X2
N, M = 300, 500
X1 = randn(N, M) / sqrt(M)
U, V = Matrix(qr(randn(N,N)).Q), Matrix(qr(randn(M,M)).Q)
D = [Diagonal((collect(range(1,stop = 0,length = N)) .- 1).^4) zeros(N, M - N)]
X2 = U*D*V'
X = [X1, X2];

# mix X1, X2 linearly
A = randn(2,2) # mixing matrix
Z = A*X

# use freecf to recover mixing matrix and free components (up to permutation and rescaling)
Aest, Xest = freecf(Z; mat = "rec") # "rec" tells the function that we are dealing with the rectangular matrices

# Aest recover A upto column permutation and column rescaling.
@show pinv(Aest)*A # their product should approximate a diagonal matrix
```

## License

This package is provided as is under the MIT License. 

## Author

Hao Wu

lingluan@umich.edu

University of Michigan, Department of Mathematics
