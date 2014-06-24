function [cost, info, x, A] = test02(n)
% function [cost, x, A] = test02(n)
% All intputs are optional.
%
% Typical call: [cost, x, A] = test2(1000);
% Perhaps the simplest test code: computes the dominant eigenvector of an
% n-by-n matrix.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    clc;
    reset(RandStream.getDefaultStream);
    randnfoo = randn(123456, 1); %#ok<NASGU>
    
    if ~exist('n', 'var') || isempty(n)
        n = 1042;
    end

    % Define the problem data
    A = randn(n);
    A = .5*(A+A');
    
    % Create the problem structure
    M = spherefactory(n);
    problem.M = M;
    
    % Define the problem cost function
%     problem.cost = @(x) -x'*(A*x);
%     problem.grad = @(x) -M.proj(x, 2*A*x);

    problem.costgrad = @(x) costgrad(A, x);
    function [cost grad] = costgrad(A, x)
        Ax = A*x;
        cost = -x'*Ax;
        if nargout == 2
            grad = -2*(Ax + cost*x);
        end
    end    

    % If the optimization algorithms require Hessians, since we do not
    % provide it, it will go for a standard approximation of it. This line
    % tells Matlab not to issue a warning when this happens.
    warning('off', 'manopt:getHessian:approx');
    
    % Check gradient consistency.
    checkgradient(problem);

    % statsfun test
    options.statsfun = @mystatsfun;
    function stats = mystatsfun(problem, x, stats)
        stats.x = x;
    end

    % stopfun test
    options.stopfun = @mystopfun;
    function stopnow = mystopfun(problem, x, info, last)
        stopnow = (last >= 3 && info(last-2).cost - info(last).cost < 1e-4);
    end
    
    % Solve
%     [x cost info] = steepestdescent(problem);
    [x cost info] = trustregions(problem, [], options);
%     [x cost info] = neldermead(problem);

% keyboard;
    
end
