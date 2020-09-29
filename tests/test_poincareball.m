function test_poincareball()

    clear all; clc; close all;
    k = 3;
    n = 10;
    manifold = poincareballfactory(k, n);
    
    problem.M = manifold;
    
    P = manifold.rand();
    problem.cost = @cost;
    function f = cost(X)
        f = norm(X - P, 'F').^2;
    end
    
    problem.grad = @(X) problem.M.egrad2rgrad(X, egrad(X));
    function g = egrad(X)
        g = 2 * (X - P);
    end
    
    problem.hess = @(X, U) problem.M.ehess2rhess(X, egrad(X), ehess(X, U), U);
    function Ress = ehess(X, eta)
        Ress = 2 * eta;
    end

% % Check numerically whether gradient and Ressian are correct
    checkgradient(problem);
    drawnow;
    pause;
    checkhessian(problem);
    drawnow;
    pause;
    
    % Initialization
    X0 = [];
    
    % Options (not mandatory)
    options.maxiter = inf;
    options.maxinner = 30;
    options.maxtime = 120;
    options.tolgradnorm = 1e-5;
    
    % Pick an algorithm to solve the problem
    [Xopt costopt info] = trustregions(problem, X0, options);
    % [Xopt costopt info] = steepestdescent(problem, X0, options);
    % [Xopt costopt info] = conjugategradient(problem, X0, options);
    
    evs = real(hessianspectrum(problem, Xopt));
    evs = real(evs);
    max(evs)/min(evs)
    stairs(sort(evs));
    title(['Eigenvalues of the Hessian of the cost function ' ...
        'at the solution']);
end