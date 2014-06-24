function [cst X A B C] = test05(m, n)
% function [cst X A B C] = test05(m, n)
% All intputs are optional.
%
% Test case for the Euclidean space (classical nonlinear optimization).
% We generate three matrices A, B and C and try to solve the Sylvester
% equation AX + XB = C by minimizing the residual ||AX+XB-C||^2.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    if ~exist('m', 'var') || isempty(m)
        m = 3;
    end
    if ~exist('n', 'var') || isempty(n)
        n = 10;
    end

    % Create the problem structure
    problem.M = euclideanfactory(m, n);

    % Define the problem data
    A = randn(m);
    B = randn(n);
    C = randn(m, n);
    
    % Define the problem cost function
    problem.cost = @(X) .5*norm(A*X+X*B-C, 'fro')^2;
    problem.grad = @(X) A'*(A*X+X*B-C) + (A*X+X*B-C)*B';
    problem.hess = @(X, H) A'*(A*H+H*B) + (A*H+H*B)*B';
    
    % Check differnetials consistency.
    checkgradient(problem);
    checkhessian(problem);

    % Solve
    [X cst] = trustregions(problem);
    
end
