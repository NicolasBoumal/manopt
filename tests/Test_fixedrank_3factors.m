function Test_fixedrank_3factors()
% function Test_LSR()
%
% Test file for the fixedrankLSRquotientfactory
% Problem: Low-rank matrix completion
% Paper link: http://arxiv.org/abs/1112.2318
%
% All intputs are optional.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, July 11, 2013.
% Contributors: Nicolas Boumal
% Change log:
    
    clc; %close all;
    
    % Define the problem data
    m = 100;
    n = 100;
    r = 5;
    A = randn(m, r);
    S = randn(n, r);
    Xopt = A*S'; % Original low-rank matrix
    
    % Create the problem structure
    % quotient LSR geometry
    
%         M = fixedrankfactory_3factors(m, n, r);
    M = fixedrankfactory_3factors_preconditioned(m, n, r);
    
    problem.M = M;
    
    
    % Which entries do we observe ?
    df = problem.M.dim();
    p = 3*df/(m*n);
    mask = rand(m, n) <= p;
    % mask = ones(m, n);
    
    
    % Define the problem cost function
    problem.cost = @cost;
    function f = cost(X)
        f = .5 * norm(mask.*((X.L*X.S*X.R') - Xopt ), 'fro')^2;
    end
    
    problem.grad = @(X) problem.M.egrad2rgrad(X, egrad(X));
    function g = egrad(X)
        P = (mask.^2) .*(X.L*X.S*X.R' - Xopt);
        g.L= P*X.R*X.S';
        g.S = X.L'*P*X.R;
        g.R = P'*X.L*X.S;
    end
    
    
    problem.hess = @(X, L) problem.M.ehess2rhess(X, egrad(X), ehess(X, L), L);
    function Hess = ehess(X, eta)
        P = (mask.^2) .* ( X.L*X.S*X.R' - Xopt );
        Pdot  = (mask.^2).*(eta.L*X.S*X.R' + X.L*eta.S*X.R' + X.L*X.S*eta.R');
        
        Hess.L = P*(X.R*eta.S' + eta.R*X.S') + Pdot*X.R*X.S';
        Hess.R = P'*(X.L*eta.S + eta.L*X.S) + Pdot'*X.L*X.S;
        Hess.S = X.L'*Pdot*X.R + eta.L'*P*X.R + X.L'*P*eta.R;
        
    end
    
    
    % % Check numerically whether gradient and Hessian are correct
    checkgradient(problem);
    drawnow;
    pause;
    checkhessian(problem);
    drawnow;
    pause;
    
    % Initial guess
    X0 = [];
    
    
    
    
    
    
    % Options (not mandatory)
    options.maxiter = inf;
    options.maxinner = 30;
    options.maxtime = 120;
    options.tolgradnorm = 1e-5;
    options.Delta_bar = min(m, n)*r;
    options.Delta0 =  min(m, n)*r/64;
    
    % Pick an algorithm to solve the problem
    [Xsol costopt info] = trustregions(problem, X0, options);
    % [Xsol costopt info] = steepestdescent(problem, X0, options);
    % [Xsol costopt info] = conjugategradient(problem, X0, options);
    
    
    evs = hessianspectrum(problem, Xsol);
    evs = real(evs);
    max(evs)/min(evs)
    stairs(sort(evs));
    title(['Eigenvalues of the Hessian of the cost function ' ...
        'at the solution']);
    
end
