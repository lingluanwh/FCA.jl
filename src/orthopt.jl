#using Optim

function OptOrtho(loss, grad, s)
    #--------------------------------------------------------------------------
    # Syntax:       Xopt = OptOrtho(loss, grad, s)
    # 
    # Input:        loss: loss function whose input is a orthogonal matrix
    #               grad: Euclidean gradient function of loss function w.r.t 
    #                       to the orthogonal matrix
    #               s: look for the optimizer in O(s) 
    #                       
    # Outputs:      Xopt: orthogonal matrix optmize loss function 
    #   
    # Description:  This function returns the opimizer of loss, whose feasible 
    #               set is O(s), using the GradientDescent method from 
    #               Manifold optimization package Optim 
    #
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------

    # use Stiefel manifold
    manif = Optim.Stiefel()
    
    # initialize a orthogonal matrix of size s-by-s
    X0 = complex(Matrix(qr(randn(s,s)).Q))
    
    # find the minimum of loss function using the 
    # gradient descent method, the Euclidean gradient 
    # is given by grad
    optimres = Optim.optimize(loss, grad, X0, 
        Optim.GradientDescent(manifold=manif); inplace = false)
    
    # extract and return the minimizer
    Xopt = Optim.minimizer(optimres)
    return real(Xopt)
end


function OptSphere(loss, grad, s)
    #--------------------------------------------------------------------------
    # Syntax:       xopt = OptSphere(loss, grad, s)
    # 
    # Input:        loss: loss function whose input is a unit vector
    #               grad: Euclidean gradient function of loss function w.r.t 
    #                       to the unit vector
    #               s: look for the optimizer in S(s - 1) 
    #                       
    # Outputs:      xopt: unit vector optmize loss function 
    #   
    # Description:  This function returns the opimizer of loss, whose feasible 
    #               set is S(s - 1), using the GradientDescent method from 
    #               Manifold optimization package Optim 
    #
    # Authors:      Hao Wu
    #               lingluanwh@gmail.com
    # 
    # Date:         Jan 20, 2019
    #--------------------------------------------------------------------------

    # use Sphere manifold
    manif = Optim.Sphere()

    # initialize a unit vector of size s
    x0 = randn(s)
    x0 = complex(x0/sqrt(x0'*x0))

    # find the minimum of loss function using the 
    # gradient descent method, the Euclidean gradient 
    # is given by grad
    optimres = Optim.optimize(loss, grad, x0, 
        Optim.GradientDescent(manifold=manif); inplace = false)

    # extract and return the minimizer
    xopt = Optim.minimizer(optimres)
    return real(xopt)
end