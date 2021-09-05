function realtest_AD2()
% Test AD for a real optimization problem on a power manifold (cell)

    % Verify that Manopt was indeed added to the Matlab path.
    if isempty(which('spherefactory'))
        error(['You should first add Manopt to the Matlab path.\n' ...
		       'Please run importmanopt.']);
    end
    
    % Verify that the deep learning tool box was installed
    assert(exist('dlarray', 'file') == 2, ['Deep learning tool box is '... 
    'needed for automatic differentiation.\n Please install the'...
    'latest version of the deep learning tool box and \nupgrade to Matlab'...
    ' R2021b if possible.'])
    
    % Generate the problem data.
    n = 100;
    A = randn(n);
    A = .5*(A+A');
    
    % Create the power manifold
    S = spherefactory(n);
    problem.M = powermanifold(S,2);
    
    % Define the problem cost function
    problem.cost  = @(X) -X{1}'*(A*X{2});
    
    % Define the gradient and the hessian via automatic differentiation
    problem = manoptAD(problem);

    % Numerically check gradient and Hessian consistency.
    figure;
    checkgradient(problem);
    figure;
    checkhessian(problem);
    
    % Solve.
    [x, xcost, info] = trustregions(problem);          %#ok<ASGLU>
    
    % Test
    ground_truth = svd(A);
    distance = abs(ground_truth(1) - (-problem.cost(x)));
    fprintf('The distance between the ground truth and the solution is %e \n',distance);

    
end