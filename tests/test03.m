function [cst X] = test03(n, m)
% function [cst X] = test03(n, m)
% All intputs are optional.
%
% Crude attempt at spreading m points over the (n-1) sphere to test the
% oblique manifold implementation. Based on April 13, 2012 notes:
%
% Maximize sum |xi-xj|^2 over all (i, j) pairs is equivalent to
% Minimize sum xi'*xj over all pairs.
% Since X'*X is a matrix with ones on the diagonal and the inner products
% xi'*xj off diagonals, summing all entries of X'*X does what we want. We
% do this by computing 1' * (X'*X) * 1, where 1 is a vector of ones of
% length m. This is easy to compute and to differentiate. The Euclidean
% gradient is X*1*1'. The geometric gradient is obtained by projection, as
% always. Note that we do not deal with the invariance of the objective
% under global rotations.
%
% This is not very good at spreading the points though ^^. It suffices to
% find a configuration such that 1 is in the kernel of X, while having X on
% the oblique manifold, to reach a global optimum.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    if ~exist('n', 'var') || isempty(n)
        n = 2;
    end
    if ~exist('m', 'var') || isempty(m)
        m = 42;
    end

    % Create the problem structure
    M = obliquefactory(n, m);
    problem.M = M;
    
    % Define the problem cost function
    w = ones(m, 1);
    problem.cost = @(X) .5*norm(X*w)^2;
    problem.grad = @(X) M.proj(X, (X*w)*w');
    
    % If the optimization algorithms require Hessians, since we do not
    % provide it, it will go for a standard approximation of it. This line
    % tells Matlab not to issue a warning when this happens.
    warning('off', 'manopt:getHessian:approx');
    
    % Check gradient consistency.
    checkgradient(problem);

    % Solve
    % [X cst] = steepestdescent(problem);
    [X cst] = trustregions(problem);
    
    % Plot the solution
    if n == 3
        plot3(X(1,:), X(2,:), X(3,:), 'r.');
        [a b c] = sphere(42);
        h = surface(a, b, c);
        set(h, 'FaceColor', [0 .3 .7]);
        set(h, 'FaceAlpha', .2);
        hold off;
        box off;
        axis off;
        axis equal;
        hold on;
    end
    
end
