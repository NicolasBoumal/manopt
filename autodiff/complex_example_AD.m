function complex_example_AD()
% A basic example that shows how to define the cost funtion for 
% optimization problems on complex manifolds.
%
% Note that automatic differentiation for complex number is not supported
% in the current version of the deep learning tool box. In order to 
% optimize on complex manifolds via AD, an alternative way is to define the 
% cost funtion using the preliminary functions listed in the file 
% functions_AD.m or define their own functions following the rules 
% described in that file. See the following as an example.
%
% See also: functions_AD.m 

% Main author: Xiaowen Jiang, August, 20, 2021
% Contributors: Nicolas Boumal
% Change log:
%

    % Verify that the deep learning tool box was installed
    assert(exist('dlarray', 'file') == 2, ['Deep learning tool box is '... 
    'needed for automatic differentiation.\n Please install the'...
    'latest version of the deep learning tool box and \nupgrade to Matlab'...
    ' 2021a if possible.'])
    
    % Generate the problem data.
    n = 100;
    A = randn(n, n) + 1i*randn(n, n);
    
    % Create the problem structure.
    S = spherecomplexfactory(n);
    M = powermanifold(S, 2);
    problem.M = M;
    
    % Define the problem cost function 
    % the original code is: problem.cost = @(x) .5*real(x{1}'*A*x{2})
    % translate it into a particular format with cfunctions in functions_AD.m
    problem.cost = @(x) .5*creal(cprod(cprod(ctransp(x{1}), A), x{2}));
    
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
    title(['Convergence of the trust-regions algorithm on the'...
        'complex sphere power manifold']);

end