function [] = test10()
% function test10()
%
% Test the fixed rank geometry with a quick and dirty low rank matrix
% completion problem.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    % Generate the data
    m = 500;
    n = 500;
    k = 10;
    L = randn(m, k);          % generate a random mxn matrix of rank k
    R = randn(n, k);
    A = L*R';
    % generate a random mask for observed entries
    P = sparse(round(.75*rand(m, n)));
    
    % Pick the manifold
    problem.M = fixedrankembeddedfactory(m, n, k);
    warning('off', 'manopt:fixedrank:exp');

    % Define the problem cost function
    problem.cost = @(X) .5*norm(P.*(X.U*X.S*X.V'-A), 'fro')^2;

    % And its gradient
    problem.egrad = @(X) P.*(X.U*X.S*X.V'-A);
    
    % Check differentials consistency.
    checkgradient(problem);

    % Solve with trust-regions and FD approximation of the Hessian
    warning('off', 'manopt:getHessian:approx');
    
    [U S V] = svds(P.*A, k);
    X.U = U; X.S = S; X.V = V;
    options.maxiter = 500;
%     options.Delta_bar = 2000;
%     options.Delta0 = options.Delta_bar / 8;
    X = trustregions(problem, X, options);
    
    fprintf('||X-A||_F = %g\n', norm(X.U*X.S*X.V' - A, 'fro'));
    
%     conjugategradient(problem);
%     steepestdescent(problem);
    
%     keyboard;
    
end
