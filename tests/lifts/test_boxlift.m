clear; clc;

n = 5;

lift = boxlift(n);

downstairs.M = lift.N;

A = randsym(n);
b = randn(n, 1);
downstairs.cost = @(x) .5*(x'*A*x) + b'*x;
downstairs.grad = @(x) A*x + b;
downstairs.hess = @(x, xdot) A*xdot;
[upstairs, downstairs] = manoptlift(downstairs, lift); % , 'AD'

% checkgradient(downstairs);
% checkhessian(downstairs);
% checkgradient(upstairs);
% checkhessian(upstairs);

[y, ~, info] = trustregions(upstairs);

x = lift.phi(y)
