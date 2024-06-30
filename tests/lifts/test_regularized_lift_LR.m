clear; clc;

m = 1000;
n = 2000;

A = sprandn(m, n, .01);

Rmn = euclideanlargefactory(m, n);
downstairs.M = Rmn;
downstairs.cost = @(X) .5*Rmn.dist(X, A)^2;
downstairs.grad = @(X) Rmn.diff(X, A);
downstairs.hess = @(X, Xdot) Xdot;

r = 4; % hard cap on rank
lift = burermonteiroLRlift(m, n, r);

lambda = 8.3;

upstairs = manoptlift(downstairs, lift, 'noAD', lambda);

Y = trustregions(upstairs);

X = Rmn.to_matrix(lift.phi(Y));

svds(X, 6)'
% log10(svd(X)')
(svds(A, 6) - lambda)'
