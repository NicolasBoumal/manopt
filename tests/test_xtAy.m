clear; clf; clc;

m = 5;
n = 10;
A = randn(m, n);

% Create the manifold M as a product of two spheres (in R^m and R^n).
elems.x = spherefactory(m);
elems.y = spherefactory(n);
manifold = productmanifold(elems);

% The cost function is defined on the product manifold M.
% Thus, a point xy is a structure with two fields: x and y.
% Likewise, tangent vectors are structures with fields x and y.
problem.M = manifold;
problem.cost = @(xy) xy.x'*A*xy.y;
problem.egrad = @(xy) struct('x', A*xy.y, 'y', A'*xy.x);
problem.ehess = @(xy, xydot) struct('x', A*xydot.y, 'y', A'*xydot.x);

xy = trustregions(problem);

fprintf('Compare these values: %g, %g.\n', ...
        getCost(problem, xy), -max(svd(A)));