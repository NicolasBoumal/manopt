clear; clf; clc;

m = 1000;
n = 1500;
r = 5;

A = sprand(m, n, .01); % random sparse matrix

Rmn = euclideanlargefactory(m, n); % mind the 'large'

fixedrk = fixedrankembeddedfactory(m, n, r); % submanifold

problem.M = fixedrk;

problem.cost = @(X) .5*Rmn.dist(X, A)^2;
problem.egrad = @(X) Rmn.diff(X, A);
problem.ehess = @(X, Xdot) fixedrk.tangent2ambient(X, Xdot);

% initialization close to zero
X0 = fixedrk.rand();
X0.S = diag(rand(r, 1)/1000);

X = trustregions(problem, X0);

% Confirm that the positive singular values of X
% in the fixedrankembeddedfactory format (U, S, V)
% match the top singular values of A.
% For efficiency, we do not form X.
svd(X.S)'
svds(A, r+1)'
