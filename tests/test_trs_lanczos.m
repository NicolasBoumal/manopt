clear; clc; clf;

H = [1 0 0; 0 1 0; 0 0 1];
g = [1; 1; 1];
Delta = 100;
localdefaults.theta = .5;
localdefaults.maxinner = 10;
localdefaults.verbosity = 10;
% The following are here for the Newton solver called below
localdefaults.maxiter_newton = 100;
localdefaults.tol_newton = 1e-16;

% Merge local defaults with user options, if any
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

[y, iter, lambda, status] = minimize_quadratic_newton(H, g, Delta, options);
disp(y);
disp(iter);
disp(lambda);
disp(status);