function complex_example_AD()
% A basic example that shows how to define the cost funtion for 
% optimization problems on complex manifolds.
%
% Note that automatic differentiation for complex numbers is not supported
% for Matlab R2021a or earlier. To fully exploit the convenience of AD,
% please update to the latest version if possible. If the user cannot have 
% access to Matlab R2021b or later, manopt provides an alternative way to 
% deal with complex problems which requires the user to define the cost 
% funtion using the basic functions listed in the folder /functions_AD or 
% to define their own functions following the rules described in that file.
% See the following as an example.
%
% See also: manoptADhelp

% This file is part of Manopt and is copyrighted. See the license file.
%
% Main author: Xiaowen Jiang, August, 31, 2021
% Contributors: Nicolas Boumal
% Change log:
%

    % Verify that the deep learning tool box was installed
    assert(exist('dlarray', 'file') == 2, ['Deep learning tool box is '... 
    'needed for automatic differentiation.\n Please install the'...
    'latest version of the deep learning tool box and \nupgrade to Matlab'...
    ' R2021b if possible.'])
    
    % Generate the problem data.
    n = 100;
    A = randn(n, n) + 1i*randn(n, n);
    A = .5*(A+A');

    % Create the problem structure.
    S = spherecomplexfactory(n);
    problem.M = S;
    
    % Define the problem cost function 
    % For Matlab R2021b or later, define the problem cost function as usual
    % problem.cost  = @(X) -.5*real(X'*A*X);
    
    % For Matlab R2021a or earlier, translate the cost function into a 
    % particular format with the basic functions in /functions_AD
    problem.cost  = @(X) -creal(cprod(cprod(ctransp(X), A), X));
    
    % Define the gradient and the hessian via automatic differentiation
    problem = manoptAD(problem);

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
    title(['Convergence of the trust-regions algorithm on the'...
        'complex sphere power manifold']);

end
