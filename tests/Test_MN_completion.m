function Test_MN_completion()
% Test for fixedrankMNquotientfactory geometry (low rank matrix completion)
%
% Paper link: http://arxiv.org/abs/1209.0068
%

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, July 11, 2013.
% Contributors: 
% Change log: 

    clear all; close all; clc;
%     reset(RandStream.getDefaultStream);
%     randnfoo = randn(1, 1); %#ok<NASGU>
    
    m = 100;
    n = 100;
    r = 5;
    A = randn(m, r);
    B = randn(n, r);
    C = A*B';
    
    % Problem structure
    % quotient UBV geometry 
    problem.M = fixedrankMNquotientfactory(m, n, r);
    
    df = problem.M.dim();
    p = 5*df/(m*n);
    mask = rand(m, n) <= p;
    
    
    problem.cost = @cost;
    function f = cost(X)
        f = .5*norm(mask.*(X.M*X.N' - C), 'fro')^2;
    end
    
problem.grad = @(X) problem.M.egrad2rgrad(X, egrad(X));
    function g = egrad(X)
        S =  mask.^2 .* (X.M*X.N' - C);
        g.M = S*X.N;
        g.N = S'*X.M;
    end

problem.hess = @(X, U) problem.M.ehess2rhess(X, egrad(X), ehess(X, U), U);
    function Hess = ehess(X, eta)
        S = mask.^2 .* (X.M*X.N' - C);
        S_star  = mask.^2 .*(eta.M*X.N' + X.M*eta.N');
        
        Hess.M = S*eta.N + S_star*X.N;
        Hess.N = S'*eta.M + S_star'*X.M;
    end


    checkgradient(problem);
    drawnow;
    pause;
    checkhessian(problem);
    drawnow;
    pause;
%     
%     problem = rmfield(problem, 'hess');

    [U S V] = svds(mask.*C, r);
    X0 = struct('M', U, 'N', V*S);

    options.statsfun = @statsfun;
    function stats = statsfun(problem, x, stats)
        stats.RMSE = nan;%norm(C - x.M*x.N', 'fro')/sqrt(m*n);
    end
    
    % Options (not mandatory)
    options.maxiter = inf;
    options.maxinner = 30;
    options.maxtime = 120;
%     options.mininner = problem.M.dim();
    options.tolgradnorm = 1e-9;
    options.Delta_bar = 1e5;
    options.Delta0 = 1e4;
%     options.useRand = true;
    

    % Pick an algorithm
    [Xopt costopt info] = trustregions(problem, X0, options);
%     [Xopt costopt info] = steepestdescent(problem, X0, options);
    
%     keyboard;

    % Plots
    subplot(3, 1, 1);
    semilogy([info.iter], [info.cost], '.-');
    subplot(3, 1, 2);
    semilogy([info.iter], [info.gradnorm], '.-');
    subplot(3, 1, 3);
    semilogy([info.iter], [info.RMSE], '.-');

end