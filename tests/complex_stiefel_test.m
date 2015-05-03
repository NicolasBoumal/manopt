function complex_stiefel_test()
    
    % Minimize the complex Brockett function
    % Generate the problem data.
    n = 100;
    p = 10;
    A = randn(n)+1i*randn(n);
    A = .5*(A+A');
    N = diag(p:-1:1);
    
    % Create the problem structure.
    manifold = complexstiefelfactory(n,p);
    problem.M = manifold;
    
    % Define the problem cost function and its Euclidean gradient.
    problem.cost  = @(x) -real(trace(x'*A*x*N));
    problem.egrad = @(x) -2*A*x*N;
    problem.ehess = @(x,d) -2*A*d*N;
    
    % Numerically check gradient consistency (optional).
    checkgradient(problem);
    checkhessian(problem);
    
    % Solve.
    [x, xcost, info, options] = trustregions(problem);
    
    % Display some statistics.
    figure;
    semilogy([info.iter], [info.gradnorm], '.-');
    xlabel('Iteration number');
    ylabel('Norm of the gradient of f');
    
end