function Test_EDM()
% function test_EDM()
%
% Low-rank Euclidean distance matrix completion
% This test file is related to the paper,
%
% B. Mishra, G. Meyer and R. Sepulchre,
% "Low-rank optimization for distance matrix completion",
% IEEE CDC, 2011
%
% Paper link: http://www.montefiore.ulg.ac.be/~mishra/papers/1112.pdf
% Fixed rank geometry: symfixedrankYYfactory
%

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, July 11, 2013.
% Contributors:
% Change log:

clear all; clc; close all;

% Problem data
m = 100;
r = 5;
Yo = randn(m,r); % True embedding
C = squareform(pdist(Yo)'.^2); % True distances

% Create the problem structure
% quotient YY geometry
problem.M = symfixedrankYYfactory(m, r);

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
    function f = cost(Y)
        YY = Y*Y';
        m = size(Y, 1);
        KvY = diag(YY)*ones(1,m)  + ones(m,1)*diag(YY)' - 2*YY;
        f = 0.5*(norm(mask.*(KvY - C), 'fro')^2);
    end

    function g = egrad(Y)
        YY = Y*Y';
        m = size(Y, 1);
        KvY = diag(YY)*ones(1,m)  + ones(m,1)*diag(YY)' - 2*YY;
        J= eye(m) - (1/m)*ones(m,1)*ones(m,1)' ;
        mat = (mask.^2).*(KvY - C);
        R = 2*J*(diag(mat*ones(m,1)) - mat)*J;
        g = 2*R*Y;
    end

    function Hess = ehess(Y, eta)
        YY = Y*Y';
        m = size(Y, 1);
        KvY = diag(YY)*ones(1,m)  + ones(m,1)*diag(YY)' - 2*YY;
        J = eye(m) - (1/m)*ones(m,1)*ones(m,1)' ;
        mat = (mask.^2).*(KvY - C);
        R = 2*J*(diag(mat*ones(m,1)) - mat)*J;
        
        Ydot = eta*Y' + Y*eta';
        YdotV = Ydot;
        KvYdot = diag(YdotV)*ones(1, m) + ones(m, 1)*diag(YdotV)' - 2*YdotV;
        mat2 = (mask.^2).*KvYdot;
        KvstarMatdot = 2*J*(diag(mat2*ones(m,1)) - mat2)*J;
        
        Hess = 2*R*eta  +  2*KvstarMatdot*Y;
    end

problem.grad = @(Y) problem.M.egrad2rgrad(Y, egrad(Y));
problem.hess = @(Y, U) problem.M.ehess2rhess(Y, egrad(Y), ehess(Y, U), U);


% % Check numerically whether gradient and Hessian are correct
% checkgradient(problem);
% drawnow;
% pause;
% checkhessian(problem);
% drawnow;
% pause;

% Initialization
[U, S, ~ ] = svds(-0.5*J*(mask.*C)*J, r); % This initialization is based on the strain error minimization of EDM
Y0 = U*(S.^0.5);


% Options (not mandatory)
options.maxiter = inf;
options.maxinner = 30;
options.maxtime = 120;
options.tolgradnorm = 1e-9;
options.Delta_bar = m * r;
options.Delta0 = options.Delta_bar / 8;

% Pick an algorithm to solve the problem
% [Yopt costopt info] = trustregions(problem, Y0, options);
% [Yopt costopt info] = steepestdescent(problem, Y0, options);
[Yopt costopt info] = conjugategradient(problem, Y0, options);



end