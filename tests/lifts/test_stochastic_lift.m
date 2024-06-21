clear; clf; clc;

n = 3;
m = 3;

lift = hadamardlift('rowstochastic', n, m);

downstairs.M = lift.N;

A = rand(n, m);
A = A ./ sum(A, 2);
sqfrobnorm = @(Z) Z(:)'*Z(:);
downstairs.cost = @(X) .5*sqfrobnorm(X*X - A);
downstairs.grad = @(X) (X*X-A)*X' + X'*(X*X-A);
downstairs.hess = @(X, Xdot) (Xdot*X + X*Xdot)*X' + ...
                             X'*(Xdot*X + X*Xdot) + ...
                             (X*X-A)*Xdot' + ...
                             Xdot'*(X*X-A);

[upstairs, downstairs] = manoptlift(downstairs, lift); % , 'AD'

% checkgradient(downstairs);
% checkhessian(downstairs);
% checkgradient(upstairs);
% checkhessian(upstairs);

[Y, ~, info] = trustregions(upstairs);

X = lift.phi(Y)
