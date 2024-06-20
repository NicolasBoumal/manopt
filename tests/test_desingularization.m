clear; clf; clc;

m = 7;
n = 9;
r = 3;
alpha = rand();

M = desingularizationfactory(m, n, r, alpha);

checkmanifold(M);

X = M.rand();
Xd = M.randvec(X);

Xd2 = M.tangent(X, Xd);
fprintf('This should be zero: %g\n', ...
        M.norm(X, M.lincomb(X, 1, Xd, -1, Xd2)));


A = randn(m, n);

Rmn = euclideanlargefactory(m, n);

problem.M = M;
problem.cost = @(X) .5*Rmn.dist(X, A)^2;
problem.egrad = @(X) Rmn.diff(X, A);
% !! Mind the ".X" at the end to extract the X part of the vector.
% The other part is the P part, which is not needed.
problem.ehess = @(X, Xdot) M.tangent2ambient(X, Xdot).X;

checkgradient(problem);
% pause;
% Use a second-order retraction to check the Hessian.
problem.M.retr = problem.M.retr_metric_proj;
checkhessian(problem);
% pause;

X = trustregions(problem);

svd(M.triplet2matrix(X))'
svd(A)'
