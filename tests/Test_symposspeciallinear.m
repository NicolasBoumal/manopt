function Test_symposspeciallinear()
% function Test_positivedefinite()
% Test for sympositivedefinite geometry (matrix completion)
%

% This file is part of Manopt: www.manopt.org.
% Original author: Alexander Müller, January 26, 2022.
% Contributors:
% Change log:

% Problem
n = 2;
B = randn(n, n)/10;
B(1,1)=B(1,1)+1;
B(2,1)=B(2,1)+1;
C = B'*B;


% Create the manifold structure
problem.M = symposspeciallinear(n);
problem.M.transp = problem.M.paralleltransp;
% problem.M.retr = problem.M.exp;

% cost description
problem.cost = @cost;
    function f = cost(X)
        f = .5*norm(C*X, 'fro')^2;
    end


% gradient description
problem.grad = @(X) problem.M.egrad2rgrad(X, egrad(X));
    function g = egrad(X)
        g = (C*C*X);
    end


%     % Hessian description
problem.hess = @(X, U) problem.M.ehess2rhess(X, egrad(X), ehess(X, U), U);
    function Hess = ehess(X, eta)
        Hess = C*C*eta;
    end

% Check numerically whether gradient and Ressian are correct
% checkretraction(problem.M);
% drawnow;
% pause;
checkgradient(problem);
drawnow;
pause;
% checkhessian(problem);
% drawnow;
% pause;

X0=eye(n);
eta0=problem.M.randvec(X0);
% trace(X0\eta0)

det(X0)

% Options (not mandatory)
options.maxiter = 20;
options.maxinner = 1000;
options.maxtime = 120;
options.tolgradnorm = 1e-10;

% Pick an algorithm to solve the problem
[Xopt, costopt, info] = trustregions(problem, X0, options);
%         [Xopt costopt info] = conjugategradient(problem, X0, options);
%         [Xopt costopt info] = steepestdescent(problem, X0, options);
Xopt
det(Xopt)
end

