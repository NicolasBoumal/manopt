function [Q A] = test08(A)
% function test80()
%
% Finds a proper rotation matrix Q such that QA = -AQ, where A is a
% skew-symmetric matrix.
%
% See personal notes of June 12, 2012
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
    
    % Generate the data
    if ~exist('A', 'var') || isempty(A)
        n = 3;
        A = randskew(n);
    else
        n = size(A, 1);
        assert(n == size(A, 2));
        assert(norm(A+A') < 1e-13);
    end
    
    % Pick the manifold
    problem.M = rotationsfactory(n);

    % Define the problem cost function
    problem.cost = @(Q) .5*norm(A*Q+Q*A, 'fro')^2;

    % And its gradient
    problem.grad = @(Q) A'*Q'*A'*Q-Q'*A*Q*A;
    
    % Check differentials consistency.
    checkgradient(problem);

    % Solve with trust-regions and FD approximation of the Hessian
    warning('off', 'manopt:getHessian:approx');
    
    options = struct();
    options.tolgradnorm = 0;
%     options.debug = true;
%     options.Delta_bar = 1.7321 * 4^3;
%     options.Delta0 = options.Delta_bar / 8;
    Q = trustregions(problem, [], options);

    return;
    
    % Add a little Nelder-Mead testing
    optionsnm.maxcostevals = 50;
    neldermead(problem, [], optionsnm);
    
end
