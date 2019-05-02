# FCA.jl

"""
Module providing the free componant analysis (FCA) algorithm.
"""
module FCA

# Package requried
using Optim # for manifold optimization
using LinearAlgebra
using GenericLinearAlgebra
using Statistics
using Combinatorics


import StatsBase: sample # for sample in embed.jl

# export functions
export  freecf, # main function
        
        # loss function
        κ₄,
        neg_abs_sum_free_kurt,
        free_ent,
        sum_free_ent,
        
        # gradient function
        grad_neg_abs_sum_free_kurt,
        grad_sum_free_ent,
        
        # freely center and whiten
        mat_center,
        free_whiten,
        
        # embed
        mat_embed,

        # icf
        icf,
        kurt,
        neg_abs_sum_kurt,
        grad_neg_abs_sum_kurt,
        negent,
        ent_sum,
        grad_ent_sum,

        # errpd
        err_pd

# include sources
include("loss.jl")
include("gradient.jl")
include("fcf.jl")
include("free_whiten.jl")
include("embed.jl")
include("icf.jl")
include("orthopt.jl")
include("errpd.jl")

end

