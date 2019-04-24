function test()
    n = 100;
    r = 5;
    
    A = randn(n,n);
    A = 0.5*(A + A');
    
    problem.M = symfixedrankpolarfactory(n, r);
    
    problem.cost = @cost;
    problem.egrad = @egrad;
    problem.ehess = @ehess;
    
    symm = @(a) 0.5*(a+a');
    
    function f = cost(x)
        f = 0.5*norm(x.U*x.B*x.U' - A, 'fro')^2;
    end
    
    function g = egrad(x)
        Res = x.U*x.B*x.U' - A;
        
        g.U = 2*Res*x.U*x.B;
        g.B = x.U'*Res*x.U;
    end
    
    function gdot = ehess(x, xdot)
        Res = x.U*x.B*x.U' - A;
        Resdot = 2*symm(xdot.U*x.B*x.U') + x.U*xdot.B*x.U';
        
        gdot.U = 2*Resdot*x.U*x.B + 2*Res*xdot.U*x.B + 2*Res*x.U*xdot.B;
        gdot.B = x.U'*Resdot*x.U + 2*symm(xdot.U'*Res*x.U);
    end
    
    checkgradient(problem);
    pause;
    checkhessian(problem);
    pause;
    
    manoptsolve(problem);
    
end