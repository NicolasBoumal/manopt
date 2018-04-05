clear all; close all; clc;

n = 5;
m = 10;

[U, ~] = qr(randn(n, min(n, m)), 0);
[V, ~] = qr(randn(m, min(n, m)), 0);
Sigma = diag(min(n,m):-1:1);
% Sigma(2,2) = Sigma(1,1);
% Sigma(3,3) = Sigma(1,1)+1e-6;
% Sigma(4,4) = Sigma(1,1);
A = U*Sigma*V';

r = 3;

problem.M = productmanifold(struct('U', stiefelfactory(n, r), ...
                                   'V', stiefelfactory(m, r)));

D = diag(r:-1:1);
problem.cost = @(X) -trace(D'*X.U'*A*X.V);
problem.egrad = @(X) struct('U', -A*X.V*D', 'V', -A'*X.U*D);
problem.ehess = @(X, Xdot) struct('U', -A*Xdot.V*D', 'V', -A'*Xdot.U*D);

opts.tolgradnorm = 1e-8;
X = trustregions(problem, [], opts);

X.Sigma = X.U'*A*X.V;

diag(X.Sigma) - svds(A, 3)

% The above works fine if the top r singular values of A are simple. It
% seems it also works when singular values are multiple; check this.

X.Sigma