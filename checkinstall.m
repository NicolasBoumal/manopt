function checkinstall()
    
    % Verify that Manopt was indeed added to the Matlab path.
    if isempty(which('spherefactory'))
        error(['You should first add Manopt to the Matlab path.\n' ...
		       'Please run importmanopt.']);
    end
    
    % Generate the problem data.
    n = 1000;
    A = randn(n);
    A = .5*(A+A');
    
    % Create the problem structure.
    manifold = spherefactory(n);
    problem.M = manifold;
    
    % Define the problem cost function and its gradient.
    problem.cost  = @(x) -x'*(A*x);
    problem.egrad = @(x) -2*A*x;
    problem.ehess = @(x, xdot) -2*A*xdot;
    
    % Numerically check gradient and Hessian consistency.
    figure(1);
    checkgradient(problem);
    figure(2);
    checkhessian(problem);
 
    % Solve.
    [x, xcost, info] = trustregions(problem);          %#ok<ASGLU>
    
    % Display some statistics.
    figure(3);
    semilogy([info.iter], [info.gradnorm], '.-');
    xlabel('Iteration #');
    ylabel('Gradient norm');
    title('Convergence of the trust-regions algorithm on the sphere');
    
end
