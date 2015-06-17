function Brockett_Wen_bfgs_test()
    
    n = 500;
    p = 4;
    B = randn(n, n); B = B + B';
    D = sparse(diag(p : -1 : 1));
    
    M = stiefelfactory(n, p);
    
    problem.M = M;
    problem.cost  = @cost;
    problem.egrad = @egrad;

    % Cost function
    function f = cost(X)
        f = trace(X'*(B*X*D));
    end
    
    % Euclidean gradient of the cost function
    function g = egrad(X)
        g = 2*B*X*D;
    end
    
    
    problem.precon = preconBFGS(problem);
    problem.linesearch = @(x, xdot, storedb, key) 1;
    options.beta_type ='steep';
    options.tolgradnorm = 1e-5;
    conjugategradient(problem, [], options);
    
    pause;
    
    % For comparison, run RTR-FD
    warning('off', 'manopt:getHessian:approx');
    problem = rmfield(problem, 'precon');
    trustregions(problem);
    
end
