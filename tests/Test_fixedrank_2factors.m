function Test_fixedrank_2factors()
% function Test_LR()
% Test for fixedrankLRquotientfactory geometry (low rank matrix completion)
%
% Paper link: http://www.icml-2011.org/papers/350_icmlpaper.pdf
%

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, July 11, 2013.
% Contributors:
% Change log:
    
    clear all; clc; close all;
    
    % Problem
    m = 20;
    n = 20;
    r = 8;
    A = randn(m, r);
    B = randn(n, r);
    C = A*B';
    
    % Create the problem structure
    
%     problem.M = fixedrankfactory_2factors(m, n, r);
    %     problem.M = fixedrankfactory_2factors_preconditioned(m, n, r);
        problem.M = fixedrankfactory_2factors_subspace_projection(m, n, r);
    
    df = problem.M.dim();
    p = 8*df/(m*n);
    %     mask = rand(m, n) <= p;
    mask = ones(m,n );
    
    problem.cost = @cost;
    function f = cost(X)
        f = .5*norm(mask.*(X.L*X.R' - C), 'fro')^2;
    end
    
    problem.grad = @(X) problem.M.egrad2rgrad(X, egrad(X));
    function g = egrad(X)
        P = mask.^2 .* (X.L*X.R' - C);
        g.L = P*X.R;
        g.R = P'*X.L;
    end
    
    problem.hess = @(X, U) problem.M.ehess2rhess(X, egrad(X), ehess(X, U), U);
    function Ress = ehess(X, eta)
        P = (mask.^2).*( X.L*X.R' - C);
        Pdot = (mask.^2).*(eta.L*X.R' + X.L*eta.R');
        Ress.L = Pdot*X.R + P*eta.R;
        Ress.R = Pdot'*X.L + P'*eta.L;
    end
    
    % % Check numerically whether gradient and Ressian are correct
    checkgradient(problem);
    drawnow;
    pause;
    checkhessian(problem);
    drawnow;
    pause;
    
    % Initialization
    X0 = [];
    
    % Options (not mandatory)
    options.maxiter = inf;
    options.maxinner = 30;
    options.maxtime = 120;
    options.tolgradnorm = 1e-5;
    options.Delta_bar = min(m ,n )*r;
    options.Delta0 = min(m ,n )*r/64;
    
    % Pick an algorithm to solve the problem
    [Xopt costopt info] = trustregions(problem, X0, options);
    % [Xopt costopt info] = steepestdescent(problem, X0, options);
    % [Xopt costopt info] = conjugategradient(problem, X0, options);
    
    evs = real(hessianspectrum(problem, Xopt));
    evs = real(evs);
    max(evs)/min(evs)
    stairs(sort(evs));
    title(['Eigenvalues of the Hessian of the cost function ' ...
        'at the solution']);
end