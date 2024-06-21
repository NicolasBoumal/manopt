clear; clf; clc;

m = 10;
n = 15;
r = 3;

lift = burermonteiroLRlift(m, n, r);
Rmn = lift.N;
downstairs.M = Rmn;

A = randn(m, n);
downstairs.cost = @(X) .5*Rmn.dist(X, A)^2;
downstairs.grad = @(X) Rmn.diff(X, A);
downstairs.hess = @(X, Xdot) Xdot;
[upstairs, downstairs] = manoptlift(downstairs, lift); % , 'AD'

% checkgradient(downstairs);
% checkhessian(downstairs);
% checkgradient(upstairs);
% checkhessian(upstairs);

[LR, ~, info] = trustregions(upstairs);

svd(Rmn.to_matrix(LR))'
svd(A)'
