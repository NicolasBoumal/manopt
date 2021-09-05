function basic_example_AD()
% A basic example that shows how to apply automatic differentiation to
% computing the gradient and the hessian.
%
% Note: Computation of the hessian is not available via AD for Matlab 
% version R2020b or earlier. To fully exploit the convenience of AD,
% please update to R2021b or later if possible.
% 
%
% See also: manoptAD, manoptADhelp

% This file is part of Manopt and is copyrighted. See the license file.
%
% Main author: Xiaowen Jiang, August, 31, 2021
% Contributors: Nicolas Boumal
% Change log:
%    

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
    
    % Create the problem structure.
    manifold = spherefactory(n);
    problem.M = manifold;
    
    % Define the problem cost function
    problem.cost  = @(x) -x'*(A*x);
    
    % Provide the gradient and the hessian via automatic differentiation
    problem = manoptAD(problem);
    
    % If the egrad has already been provided, the ehess will be computed 
    % according to the egrad, which maybe faster based on the expression 
    % of the egrad.
    % problem.egrad = @(x) -2*(A*x);
    % problem = manoptAD(problem);

    % If the user only wants the gradient or the hessian information,
    % set the second argument of manoptAD to be 'egrad' or 'ehess'

    % e.g. Provide the gradient only and use FD approximation of hessian
    % (which is often faster than providing the exact hessian).
    % problem = manoptAD(problem,'egrad');

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
