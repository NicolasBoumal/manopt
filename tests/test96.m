function test96()
% function test96()
% Low-rank Euclidena distance matrix completion
%
% We use the tuned geometry, symfixedrankNewYYquotientfactory to solve the
% EDMCP. This test file is different from test file 'test97'
% 

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Dec. 30, 2012.
% Contributors: 
% Change log: 


clear all; clc; close all;

m = 500;
r = 5;
Yo = randn(m,r); % True embedding
C = squareform(pdist(Yo)'.^2); % True distances

% Create the problem structure
% quotient YY (tuned for least square problems) geometry
problem.M = symfixedrankNewYYquotientfactory(m,  r);

df = problem.M.dim();
p = 5*df/(m*m);
symm = @(M) .5*(M+M');

% mask = symm(rand(m, m)) <= p;
mask = spones(sprandsym(m, p));

prop_known = sum(sum(mask == 1))/(m*m);
fprintf('Low-rank EDM completion... \n');
fprintf('Fraction of entries given: %f \n', full(prop_known));

J = eye(m) - (1/m)*ones(m,1)*ones(m,1)'; % Centering matrix

problem.cost = @cost;
    function f = cost(X)
        YY = X.Y*X.Y';
        m = size(X.Y, 1);
        KvX = diag(YY)*ones(1,m)  + ones(m,1)*diag(YY)' - 2*YY;
        f = 0.5*(norm(mask.*(KvX - C), 'fro')^2);
    end

problem.grad = @grad;
    function g = grad(X)
        Y = X.Y;
        YY = X.Y*X.Y';
        m = size(X.Y, 1);
        KvX = diag(YY)*ones(1,m)  + ones(m,1)*diag(YY)' - 2*YY; 
        J= eye(m) - (1/m)*ones(m,1)*ones(m,1)' ;
        mat = mask.*(KvX - C);
        R = J*(diag(mat*ones(m,1)) - mat)*J; 
        R = 4*R;
        r = size(Y, 2);
        YtY = Y'*Y;
        invYtY = eye(r) / YtY;
        
        g = struct('Y', R*Y*invYtY);
    end

problem.hess = @hess;
    function Hess = hess(X, eta)
        
        Y = X.Y;
        YY = X.Y*X.Y';
        m = size(X.Y, 1);
        KvX = diag(YY)*ones(1,m)  + ones(m,1)*diag(YY)' - 2*YY; 
        J = eye(m) - (1/m)*ones(m,1)*ones(m,1)' ;
        mat = mask.*(KvX - C);
        R = J*(diag(mat*ones(m,1)) - mat)*J; 
        R = 4*R;

        Xdot = eta.Y*Y' + Y*eta.Y';
        XdotV = Xdot;
        KvXdot = diag(XdotV)*ones(1, m) + ones(m, 1)*diag(XdotV)' - 2*XdotV;
        mat2 = mask.*KvXdot;
        KvstarMatdot = J*(diag(mat2*ones(m,1)) - mat2)*J;
        KvstarMatdot = 4*KvstarMatdot;
        
        r = size(Y, 2);
        YtY = Y'*Y;
        invYtY = eye(r) / YtY;

        Hess.Y = R*eta.Y*invYtY  +  KvstarMatdot*Y*invYtY;
        Hess.Y = Hess.Y - 2*R*Y*(invYtY * symm(eta.Y'*X.Y) * invYtY);
        
        gradY = R*Y*invYtY;
        
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
[U, S, ~ ] = svds(-0.5*J*(mask.*C)*J, r); % This initialization is based on the strain error minimization of EDM
Y0 = U*(S.^0.5);
X0 = struct('Y', Y0);

% Options (not mandatory)
options.maxiter = inf;
options.maxinner = 30;
options.maxtime = 120;
options.tolgradnorm = 1e-9;
options.Delta_bar = m * r;
options.Delta0 = options.Delta_bar / 8;

% Pick an algorithm to solve the problem
   [Xopt, costopt, info] = trustregions(problem, X0, options);
%  [Xopt costopt info] = steepestdescent(problem, X0, options);
% [Xopt costopt info] = conjugategradient(problem, X0, options);



end