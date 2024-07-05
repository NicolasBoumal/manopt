clear; clc;

n = 8;
A = randn(n, n);
B = randn(n, n);
C = randn(n, n);

SOn = rotationsfactory(n);
problem.M = SOn;
problem.cost = @(X) .5*norm(A*X*B - C, 'fro')^2; % f(X)
problem.egrad = @(X) A'*(A*X*B - C)*B';          % grad f(X) in R^(nxn)

% Wrong.
% This code treats the second input of ehess as if it were in
% the embedding space. Optimization algorithms won't be amused.
problem.ehess = @(X, Xdot) A'*(A*Xdot*B)*B';

% Correct.
% This code notes that the second input to ehess is a representation of
% a tangent vector. It transforms Omg to X*Omg, which is the direction
% that is actually encoded by Omg, and which we often call Xdot.
% Equivalently, we can call SOn.tangent2ambient(X, Omg) instead of X*Omg.
problem.ehess = @(X, Omg) A'*(A*X*Omg*B)*B';

figure(1); clf;
checkgradient(problem);

figure(2); clf;
checkhessian(problem);

X = trustregions(problem);
