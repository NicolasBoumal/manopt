% NB Nov. 24, 2020
% Using a first-order retraction, aims to compute the metric projection
% retraction for a Riemannian submanifold of a Euclidean space, thus
% effectively boosting the first-order retractions to a second-order one.
%
% See also test_retraction_boost.m
clear; close all; clc;

% The default retraction on Stiefel is only first order.
M = stiefelfactory(1000, 25);
X = M.rand();
scale = 1;
V = scale * M.randvec(X);

problem.M = M;
problem.cost = @(Y) .5*norm(X+V - Y, 'fro')^2;
problem.egrad = @(Y) Y-(X+V);
problem.ehess = @(Y, Ydot) Ydot;

% Seems to need a fairly consistent 1+2+4 inner iterations, with trust
% region radius not budging... could almost hard-code it, except it
% wouldn't make much sense of course, because the observation is only for
% Stiefel here, and we know how to compute second-order retractions faster
% than with 7 QR factorizations. Observation was for unit-norm V: smaller
% vectors are cheaper. E.g., with V of norm .001, gets from 1e-8 to 1e-16
% accuracy in 1 inner iteration, so, with the Cauchy step.
Y = trustregions(problem, M.retr(X, V), struct('tolgradnorm', 1e-12));

% For Stiefel, we also have access to the metric projection retraction.
Z = M.retr_polar(X, V);
norm(Z - Y, 'fro')                  % close to zero
norm(multiskew(Y'*(X+V-Y)), 'fro')  % close to zero (specific to Stiefel)
