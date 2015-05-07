function complex_stiefel_test()
    
    % Minimize the complex Brockett function
    % Generate the problem data.
    n = 100;
    p = 10;
    A = randn(n)+1i*randn(n);
    A = .5*(A+A');
    N = diag(p:-1:1);
    
    % Create the problem structure.
    manifold = stiefelcomplexfactory(n, p);
    problem.M = manifold;
    
    % Define the problem cost function and its Euclidean gradient.
    problem.cost  = @(X) -real(trace(X'*A*X*N));
    problem.egrad = @(X) -2*A*X*N;
    problem.ehess = @(X, Xdot) -2*A*Xdot*N;
    
    % Numerically check gradient consistency (optional).
    % checkgradient(problem); pause;
    % checkhessian(problem); pause;
    
    % Solve.
    [X, Xcost, info, options] = trustregions(problem); %#ok<NASGU,ASGLU>
    
    % Display some statistics.
    figure;
    semilogy([info.iter], [info.gradnorm], '.-');
    xlabel('Iteration number');
    ylabel('Norm of the gradient of f');
    
end
