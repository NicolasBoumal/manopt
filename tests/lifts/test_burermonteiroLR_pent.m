clear; clc;


m = 1000;
n = 1500;
r = 5;

A = sprand(m, n, .01); % random sparse matrix

Rmn = euclideanlargefactory(m, n); % mind the 'large'
desing = desingularizationfactory(m, n, r); % (X, P) lift

problem.M = desing;
problem.cost = @(X) .5*Rmn.dist(X, A)^2;
problem.egrad = @(X) Rmn.diff(X, A);      % mind the 'e'
problem.ehess = @(X, Xdot) desing.tangent2ambient(X, Xdot).X; % !!




% initialization close to zero
X0 = desing.rand();
X0.S = diag(rand(r, 1)/1000);

X = trustregions(problem, X0);

% Confirm that the positive singular values of X
% in the desingularizationfactory format (U, S, V)
% match the top singular values of A.
% For efficiency, we do not form X.
svd(X.S)'
svds(A, r+1)'