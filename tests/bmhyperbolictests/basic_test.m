clear all; close all; clc;
n = 5;
m = 15;
trnsp = true;
M = hyperbolicfactory(n, m, trnsp);
problem.M = M;
A = randn(size(problem.M.rand()));
problem.cost = @(X) .5*norm(X - A, 'fro')^2;
problem.egrad = @(X) X-A;
problem.ehess = @(X, Xdot) Xdot;
checkgradient(problem); pause;
checkhessian(problem); pause;

X = M.rand();
U = M.randvec(X);