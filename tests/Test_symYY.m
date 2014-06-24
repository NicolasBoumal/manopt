function Test_symYY()
% function test98()
% Test for symfixedrankYYquotientfactory geometry (low-rank PSD matrix completion)
%
% Paper link: http://www.di.ens.fr/~fbach/journee2010_sdp.pdf
%

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, July 11, 2013.
% Contributors:
% Change log:


% We know about this warning, so it is safe to turn it off for now.
warning('off', 'manopt:symfixedrankYYfactory:exp');

clc; close all;

% Problem data
n = 1000;
r = 5;
Y_org = randn(n, r);
A = Y_org*Y_org';

% Create the problem structure
% quotient YY geometry
problem.M = symfixedrankYYfactory(n, r);

df = problem.M.dim();
p = 3.5*df/(n*n);
mask = spones(sprandsym(n, p));
prop_known = sum(sum(mask == 1))/(n*n);
fprintf('Fraction of entries given: %f \n', full(prop_known));


problem.cost = @(Y) norm(mask.*(Y*Y' - A),'fro')^2;

egrad = @(Y) 4*((mask.^2).*(Y*Y' - A))*Y;
ehess = @(Y, U) 4*((mask.^2) .* (Y*Y' - A))*U + 4*((mask.^2) .* (Y*U' + U*Y'))*Y;

problem.grad = @(Y) problem.M.egrad2rgrad(Y, egrad(Y));
problem.hess = @(Y, U) problem.M.ehess2rhess(Y, egrad(Y), ehess(Y, U), U);

% % Check numerically whether gradient and Hessian are correct
%     checkgradient(problem);
%     drawnow;
%     pause;
%     checkhessian(problem);
%     drawnow;
%     pause;

% Initialization
[U0, S0, ~ ] = svds(mask.*A, r);
Y0 = U0*(S0.^0.5);

% Options (not mandatory)
options.maxiter = inf;
options.maxinner = 30;
options.maxtime = 120;
options.tolgradnorm = 1e-5;
options.Delta_bar = n * r;
options.Delta0 = options.Delta_bar / 8;

% Pick an algorithm to solve the problem
[Yopt costopt info] = trustregions(problem, Y0, options);
% [Yopt costopt info] = steepestdescent(problem, Y0, options);
% [Yopt costopt info] = conjugategradient(problem, Y0, options);


end