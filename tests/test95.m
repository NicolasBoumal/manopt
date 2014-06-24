function test95()
% function test95()
%
% Test for symfixedrankNewYYquotientfactory geometry (low-rank PSD matrix completion)
% This test is different from 'test98'. We use the tuned geometry,
% symfixedrankNewYYquotientfactory.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Dec. 30, 2012.
% Contributors: 
% Change log: 

clear all; clc; close all;

m = 500;
r = 5;
A = randn(m, r);
C = A*A';

problem.M = symfixedrankNewYYquotientfactory(m, r);

df = problem.M.dim();
p = 5*df/(m*m);
symm = @(M) .5*(M+M');

% mask = symm(rand(m, m)) <= p;
mask = spones(sprandsym(m, p));

prop_known = sum(sum(mask == 1))/(m*m);

fprintf('Fraction of entries given: %f \n', full(prop_known));

problem.cost = @cost;
    function f = cost(X)
        f = 0.5*(norm(mask.*(X.Y*X.Y' - C), 'fro')^2);
    end

problem.grad = @grad;
    function g = grad(X)
        Y = X.Y;
        S = 2*mask.^2 .* (Y*Y' - C);
        r = size(Y, 2);
        YtY = Y'*Y;
        invYtY = eye(r) / YtY;
        g = struct('Y', S*Y*invYtY);
    end

problem.hess = @hess;
    function Hess = hess(X, eta)        
        Y = X.Y;
        S = 2*mask.*( X.Y*X.Y' - C);
        S_star  = 2*mask.*(eta.Y*X.Y' + X.Y*eta.Y');
        
        r = size(Y, 2);
        YtY = Y'*Y;
        invYtY = eye(r) / YtY;
        
        Hess.Y = S_star*X.Y*invYtY;
        Hess.Y = Hess.Y + S*eta.Y*invYtY; 
        Hess.Y = Hess.Y - 2*S*Y*(invYtY * symm(eta.Y'*X.Y) * invYtY);
       
        gradY = S*Y*invYtY;
        
        % I still need a correction factor for the non-constant metric
        Hess.Y = Hess.Y + gradY*symm(eta.Y'*X.Y)*invYtY + eta.Y*symm(gradY'*X.Y)*invYtY - X.Y*symm(eta.Y'*gradY)*invYtY;
        
        Hess = problem.M.proj(X, Hess);
        
    end

% % Check numerically whether gradient and Hessian are correct
%     checkgradient(problem);
%     drawnow;
%     pause;
%     checkhessian(problem);
%     drawnow;
%     pause;

% Initialization
[U, S, ~ ] = svds(mask.*C, r);
Y0 = U*(S.^0.5);
X0 = struct('Y', Y0);


% Options (not mandatory)
options.maxiter = inf;
options.maxinner = 30;
options.maxtime = 120;
options.tolgradnorm = 1e-9;
options.Delta_bar = m * r;
options.Delta0 = options.Delta_bar / 4;


% Pick an algorithm to solve the problem
[Xopt costopt info] = trustregions(problem, X0, options);
% [Xopt costopt info] = steepestdescent(problem, X0, options);
% [Xopt costopt info] = conjugategradient(problem, X0, options);

end