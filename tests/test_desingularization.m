clear all; close all; clc;

addpath('../examples')

m = 201;
n = 199;
r = 9;
alpha = rand();

M = desingularizationfactory(m, n, r, alpha);

checkmanifold(M);

X = M.rand();
Xd = M.randvec(X);

Xd2 = M.tangent(X, Xd);
M.norm(X, M.lincomb(X, 1, Xd, -1, Xd2))

problem = desingularization_matrix_completion();

checkgradient(problem);
pause;
% Use a second-order retraction to check the Hessian.
problem.M.retr = problem.M.retr_metric_proj;
checkhessian(problem);
pause;

trustregions(problem);
