"""
    OptOrtho(loss, grad, s)

This function returns the opimizer of loss, whose feasible 
set is O(s), using the GradientDescent method from 
Manifold optimization package Optim 

# Arguments
-   `loss`: loss function whose input is a orthogonal matrix
-   `grad`: Euclidean gradient function of loss function w.r.t 
    to the orthogonal matrix
-   `s`: The feasible set is O(s)
-   `opt_config`: configuration passed to Optim

# Outputs:      
-   `Xopt`: orthogonal matrix optmize loss function 
"""
function OptOrtho(loss, grad, s; opt_config=nothing)
    # default optim config
    if opt_config === nothing
        opt_config = Optim.Options(f_tol = 0.0,
                                    g_tol = 1e-8,
                                    iterations = 1_000)
    end

    # use Stiefel manifold
    manif = Optim.Stiefel()
    
    # initialize a orthogonal matrix of size s-by-s
    X0 = complex(Matrix(qr(randn(s,s)).Q))
    
    # find the minimum of loss function using the 
    # gradient descent method, the Euclidean gradient 
    # is given by grad
    optimres = Optim.optimize(loss, grad, X0, 
        Optim.GradientDescent(manifold=manif), opt_config; inplace = false)
    
    # extract and return the minimizer
    Xopt = Optim.minimizer(optimres)
    return real(Xopt)
end

"""
    OptOrtho(loss, grad, s)

This function returns the opimizer of loss, whose feasible 
set is S(s - 1), using the GradientDescent method from 
Manifold optimization package Optim 

# Arguments
-   `loss`: loss function whose input is a orthogonal matrix
-   `grad`: Euclidean gradient function of loss function w.r.t 
    to the orthogonal matrix
-   `s`: The feasible set is S(s-1)
-   `opt_config`: configuration passed to Optim
                      
# Outputs:      
-   `Xopt`: orthogonal matrix optmize loss function 
"""
function OptSphere(loss, grad, s; opt_config=nothing)
    # default optim config
    if opt_config === nothing
        opt_config = Optim.Options(f_tol = 0.0,
                                    g_tol = 1e-8,
                                    iterations = 1_000)
    end

    # use Sphere manifold
    manif = Optim.Sphere()

    # initialize a unit vector of size s
    x0 = randn(s)
    x0 = complex(x0/sqrt(x0'*x0))

    # find the minimum of loss function using the 
    # gradient descent method, the Euclidean gradient 
    # is given by grad
    optimres = Optim.optimize(loss, grad, x0, 
        Optim.GradientDescent(manifold=manif), opt_config; inplace = false)

    # extract and return the minimizer
    xopt = Optim.minimizer(optimres)
    return real(xopt)
end