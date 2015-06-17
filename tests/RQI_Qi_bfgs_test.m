function RQI_Qi_bfgs_test()

    n = 12;
    p = 7;
    A = randn(n, n);
    B = randn(p, p);
    
    
    M = stiefelfactory(n, p);
    
    problem.M = M;
    problem.cost  = @cost;
    problem.egrad = @egrad;

    % Cost function
    function f = cost(X)
        f = 0.5*norm(A*X - X*B, 'fro')^2;
    end
    
    % Euclidean gradient of the cost function
    function g = egrad(X)
        g = A'*(A*X - X*B) - (A*X - X*B)*B';
    end
    
    
    problem.precon = preconBFGS(problem);
    problem.linesearch = @(x, xdot, storedb, key) 1;
    options.beta_type = 'steep';
    conjugategradient(problem, [], options);
    
end
