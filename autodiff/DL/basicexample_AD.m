function basicexample_AD()
    
    % Verify that Manopt was indeed added to the Matlab path.
    if isempty(which('spherefactory'))
        error(['You should first add Manopt to the Matlab path.\n' ...
		       'Please run importmanopt.']);
    end
    
    % Verify that the deep learning tool box was installed
    assert(exist('dlarray', 'file') == 2, ['Deep learning tool box is '... 
    'needed for automatic differentiation.\n Please install the'...
    'latest version of the deep learning tool box and \nupgrade to Matlab'...
    ' 2021a if possible.'])
    
    % Generate the problem data.
    n = 1000;
    A = randn(n);
    A = .5*(A+A');
    
    % Create the problem structure.
    manifold = spherefactory(n);
    problem.M = manifold;
    
    % Define the problem cost function
    problem.cost  = @(x) -x'*(A*x);
    
    % Define the gradient and the hessian via automatic differentiation
    problem = preprocessAD(problem);
    
    % Numerically check gradient and Hessian consistency.
    figure;
    checkgradient(problem);
    figure;
    checkhessian(problem);
 
    % Solve.
    [x, xcost, info] = trustregions(problem);          %#ok<ASGLU>
    
    % Display some statistics.
    figure;
    semilogy([info.iter], [info.gradnorm], '.-');
    xlabel('Iteration #');
    ylabel('Gradient norm');
    title('Convergence of the trust-regions algorithm on the sphere');
    
end