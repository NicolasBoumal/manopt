function [Q E1 E2] = test09()
% function test09()
%
% See personal notes of June 13, 2012
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    % Generate the data
    n = 4;
    i = 1; j = 2;
    k = 3; l = 4;
    
    E1 = zeros(n);
    E1(i, j) =  1;
    E1(j, i) = -1;
    E2 = zeros(n);
    E2(k, l) =  1;
    E2(l, k) = -1;
    
    % Pick the manifold
    problem.M = rotationsfactory(n);

    % Define the problem cost function
    problem.cost = @(Q) .5*norm(E1*Q-Q*E2, 'fro')^2 + .5*norm(E2*Q+Q*E1, 'fro')^2;

    % And its gradient
    problem.grad = @(Q) problem.M.egrad2rgrad(Q, (E1'*(E1*Q-Q*E2) - (E1*Q-Q*E2)*E2' + E2'*(E2*Q+Q*E1) + (E2*Q+Q*E1)*E1'));
    
    % Check differentials consistency.
    checkgradient(problem);

    % Solve with trust-regions and FD approximation of the Hessian
    warning('off', 'manopt:getHessian:approx');
    
    Q = trustregions(problem);
    
    keyboard;
    
end