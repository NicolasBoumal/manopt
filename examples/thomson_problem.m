% This script is an example of how Manopt can approximate the gradient and
% even the Hessian of a cost function based on finite differences, if only
% the cost function is specified without its derivatives.
%
% This functionality is provided only as a help for prototyping, and should
% not be used to compare algorithms in terms of computation time or
% accuracy.
%
% See also the derivative free solvers for an alternative:
% pso and neldermead.

% Define the Thomson problem with 1/r^2 potential. That is: find n points
% x_i on a sphere in R^3 such that the sum over all pairs (i, j) of the
% potentials 1/||x_i - x_j||^2 is minimized. Since the points are on a
% sphere, each potential is .5/(1-x_i'*x_j).
n = 50;
problem.M = obliquefactory(3, n);
problem.cost = @(X) sum(sum(triu(1./(1-X'*X), 1))) / n^2;

% Attempt to minimize the cost. Since the gradient is not provided, it is
% approximated with finite differences. This is /slow/, since for each
% gradient approximation, problem.M.dim()+1 calls to the cost function are
% necessary.
%
% Note that it is difficult to reach high accuracy critical points with an
% approximate gradient, hence the need to set a less ambitious value for
% the gradient norm tolerance.
opts.tolgradnorm = 1e-3;
X = conjugategradient(problem, [], opts);

% Plot the points on a translucid sphere
[x, y, z] = sphere(50);
surf(x, y, z, 'FaceAlpha', .5);
hold all;
plot3(X(1, :), X(2, :), X(3, :), '.', 'MarkerSize', 20);
axis equal;

% For much better performance, after an early prototyping phase, the
% gradient of the cost function should be specified, typically in
% problem.grad or problem.egrad. See the online document at
%
%   http://www.manopt.org
%
% for more information.
