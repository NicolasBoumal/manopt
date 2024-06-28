clear; clf; clc;

m = 5;
n = 7;

A = randn(m, n);

Rmn = euclideanlargefactory(m, n);
downstairs.M = Rmn;
downstairs.cost = @(X) .5*Rmn.dist(X, A)^2;
downstairs.grad = @(X) Rmn.diff(X, A);
downstairs.hess = @(X, Xdot) Xdot;

r = 4; % hard cap on rank
lift = burermonteiroLRlift(m, n, r);

% This should be created by the lift itself, so that it would be consistant
% with the lift's domain and the embedded flag, without separating them.
lambda = 1.1354;
lift.rho = @(Y) (lambda/2)*(norm(Y.L, 'fro')^2 + norm(Y.R, 'fro')^2);
lift.gradrho = @(Y) struct('L', lambda*Y.L, 'R', lambda*Y.R);
lift.hessrho = @(Y, Ydot) struct('L', lambda*Ydot.L, 'R', lambda*Ydot.R);

upstairs = manoptlift_with_regularizer(downstairs, lift, 'noAD');

Y = trustregions(upstairs);

X = Rmn.to_matrix(lift.phi(Y));

svd(X)'
log10(svd(X)')
