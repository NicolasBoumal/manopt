function Test_fixedrank_elliptope()
% Test_fixedrank_spectahedron()
% The problem being considered here is the fixed-rank Maxcut
% All intputs are optional.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, July 11, 2013.
% Contributors: Nicolas Boumal
% Change log:
    
    clear all; clc; close all;
    
    % Define the problem data
    % Synthetic dataset
    n = 100;
    p = 2;
    A = randn(n, p);
    A = -A*A'; % original negative semidefinite matrix
    
    r = p; % Rank of the solution
    
    % Create the problem structure
    M = elliptopefactory(n, r);
    problem.M = M;
    
    % Define the problem cost function
    problem.cost = @cost;
    function f = cost(Y)
        AY = A*Y;
        f = trace(Y'*AY);
    end
    
    problem.grad = @(X) problem.M.egrad2rgrad(X, egrad(X));
    function g = egrad(Y)
        AY = A*Y;
        g = 2*AY;
    end
    
    
    problem.hess = @(Y, Z) problem.M.ehess2rhess(Y, egrad(Y), ehess(Y, Z), Z);
    function Hess = ehess(Y, eta)
        Hess = 2*(A*eta);
    end
    
    % Check numerically whether gradient and Hessian are correct
    checkgradient(problem);
    drawnow;
    pause;
    checkhessian(problem);
    drawnow;
    pause;
    
    % Initial guess
    
    Y0 = [];
    
    
    % Options (not mandatory)
    options.maxiter = inf;
    options.maxinner = 30;
    options.maxtime = 20;
    options.tolgradnorm = 1e-5;
    options.Delta_bar = n*r;
    options.Delta0 =  n*r/64;
    
    % Pick an algorithm to solve the problem
    [Ysol, costopt, info] = trustregions(problem, Y0, options);
    %     [Ysol costopt info] = steepestdescent(problem, Y0, options);
    %     [Ysol costopt info] = conjugategradient(problem, Y0, options);
    
    evs = hessianspectrum(problem, Ysol);
    evs = real(evs);
    max(evs)/min(evs)
    stairs(sort(evs));
    title(['Eigenvalues of the Hessian of the cost function ' ...
        'at the solution']);
    
end
