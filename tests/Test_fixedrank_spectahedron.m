function Test_fixedrank_spectahedron()
% Test_fixedrank_spectahedron()
% The problem being considered here is the fixed-rank DSPCA
% All intputs are optional.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, July 11, 2013.
% Contributors: Nicolas Boumal
% Change log:
    
    clear all; clc; close all;

    
    % Define the problem data
    n = 100;           % number of variables
    p = 100;           % number of samples
    A = randn(p,n);   % data matrix
    rho = 5;          % sparsity weight factor
    eps = 2e-4;       % smoothing parameter
    
    r = 1; % Rank of the solution
    
    % Create the problem structure
    M = spectrahedronfactory(n, r);
    problem.M = M;
    
    % Define the problem cost function
    problem.cost = @cost;
    function f = cost(Y)
        AY = A*Y;
        f = -trace(AY'*AY) + rho*sum(sum(((Y*Y').^2 + eps^2).^(0.5)));
    end
    
    problem.grad = @(X) problem.M.egrad2rgrad(X, egrad(X));
    function g = egrad(Y)
        YY = Y*Y';
        M = 2*((YY).^2 + eps^2).^(-0.5).*(Y*Y');
        g = -2*A'*(A*Y) + rho*(M*Y);
    end
    
    
    problem.hess = @(Y, Z) problem.M.ehess2rhess(Y, egrad(Y), ehess(Y, Z), Z);
    function Hess = ehess(Y, eta)
        YY = Y*Y';
        YY_sq = YY.^2;
        YY_sq_eps0 = YY_sq + eps^2;
        YY_sq_eps = YY_sq_eps0.^(-0.5);
        M = 2*YY_sq_eps.*(YY);
        Yeta = Y*eta';
        M_dot1= 2*YY_sq_eps.*(Yeta' + Yeta) ;
        M_dot2= -2*(YY_sq_eps0.^(-3/2)).*YY_sq.*(Yeta' + Yeta) ;
        M_dot = M_dot1 + M_dot2;
        Hess = -2*A'*(A*eta) + rho*(M*eta + M_dot*Y);
        
    end
    
    % Check numerically whether gradient and Hessian are correct
    checkgradient(problem);
    drawnow;
    pause;
    checkhessian(problem);
    drawnow;
    pause;
    
    % Initial guess
    [~, sig, V] = svds(A, r, 'L'); % initialization by PCA
    Y0 = V*(sig .^ 0.5);
    Y0 = Y0/norm(Y0,'fro');
    
    
    % Options (not mandatory)
    options.maxiter = inf;
    options.maxinner = 30;
    options.maxtime = 200;
    options.tolgradnorm = 1e-5;
    options.Delta_bar = n*r;
    options.Delta0 =  n*r/(4^4);
    
    % Pick an algorithm to solve the problem
    [Ysol costopt info] = trustregions(problem, Y0, options);
    %     [Ysol costopt info] = steepestdescent(problem, Y0, options);
    %     [Ysol costopt info] = conjugategradient(problem, Y0, options);
    
%     evs = hessianspectrum(problem, Ysol);
%     evs = real(evs);
%     max(evs)/min(evs)
%     stairs(sort(evs));
%     title(['Eigenvalues of the Hessian of the cost function ' ...
%         'at the solution']);
    
end
