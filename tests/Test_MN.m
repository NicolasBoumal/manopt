function Test_MN()
% function test15()
% Test for fixedrankMNquotientfactory geometry (low rank approximation)
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

% Problem data
m = 100;
n = 100;
p = 5;
A = randn(m, n);

% Fixed-rank manifold geometry
% quotient MN geometry
problem.M = fixedrankMNquotientfactory(m, n, p);

problem.cost = @(X) .5*norm(X.M*X.N' - A, 'fro')^2;

problem.grad = @(X) problem.M.egrad2rgrad(X, egrad(X));
    function g = egrad(X)
        S =  X.M*X.N' - A;
        g.M = S*X.N;
        g.N = S'*X.M;
    end

problem.hess = @(X, U) problem.M.ehess2rhess(X, egrad(X), ehess(X, U), U);
    function Hess = ehess(X, eta)
        S = ( X.M*X.N' - A);
        S_star  = (eta.M*X.N' + X.M*eta.N');
        
        Hess.M = S*eta.N + S_star*X.N;
        Hess.N = S'*eta.M + S_star'*X.M;
    end

% % Check numerically whether gradient and Hessian are correct
%     checkgradient(problem);
%     drawnow;
%     pause;
%     checkhessian(problem);
%     drawnow;
%     pause;
%
%     problem = rmfield(problem, 'hess');

[U S V] = svd(A);
Xsol.M = U(:, 1:p);
Xsol.N = V(:, 1:p)*S(1:p, 1:p);

%     X0 = problem.M.exp(Xsol, problem.M.randvec(Xsol), 1e+2);
X0 = problem.M.rand();

options.statsfun = @statsfun;
    function stats = statsfun(problem, x, stats)
        stats.dist = norm(Xsol.M*Xsol.N' - x.M*x.N', 'fro');
    end

% Options (not mandatory)
options.maxiter = 150;
options.maxinner = problem.M.dim();
%     options.mininner = problem.M.dim();
options.tolgradnorm = 1e-10;
%     options.useRand = true;


% Pick an algorithm to solve the problem
[Xopt costopt info] = trustregions(problem, X0, options);
%     [Xopt costopt info] = steepestdescent(problem, X0, options);
%     [Xopt costopt info] = conjugategradient(problem, X0, options);


%     keyboard;
subplot(3, 1, 1);
plot([info.iter], [info.cost], '.-');
subplot(3, 1, 2);
semilogy([info.iter], [info.gradnorm], '.-');
subplot(3, 1, 3);
semilogy([info.iter], [info.dist], '.-');

assert(norm(Xopt.M'*Xopt.M - eye(p)) < 1e-12);

end