function test_poincareball()

    k = 3;
    n = 10;
    manifold = poincareballfactory(k, n);
    
    problem.M = manifold;
    
    P = manifold.rand();
    problem.cost = @(X) .5*norm(X-P, 'fro').^2;
    problem.egrad = @(X) X-P;
    problem.ehess = @(X, U) U;

    % Check numerically whether gradient and Ressian are correct
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
    options.tolgradnorm = 1e-10;
    
    % Pick an algorithm to solve the problem
    Xopt = trustregions(problem, X0, options);
    % [Xopt, costopt, info] = steepestdescent(problem, X0, options);
    % [Xopt, costopt, info] = conjugategradient(problem, X0, options);
    
    % Curiously enough, the optimizer sometimes doesn't find the solution
    % P. When this happens, the erroneous columns of Xopt tend to have norm
    % very close to 1, which is also where numerics become tricky.
    Xopt - P
    sum(Xopt.^2)
    sum(P.^2)
    
    evs = hessianspectrum(problem, Xopt);
    evs = real(evs);
    stairs(sort(evs));
    title(['Eigenvalues of the Hessian of the cost function ' ...
           'at the solution']);
    fprintf('Hessian condition number at solution: %g\n', ...
            max(abs(evs))/min(abs(evs)));
end
