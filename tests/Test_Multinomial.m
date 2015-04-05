function  Test_Multinomial()
    % This file is part of Manopt: www.manopt.org.
    % Original author: Bamdev Mishra, April 06, 2015.
    % Contributors:
    % Change log:
    
    clear all; clc; close all;
    
    % Synthetic data
    m = 1000;
    n = 5;
    
    % Synthetic example
    A = rand(n, m);
    A = A./(ones(n,1)*sum(A, 1));
    A_org = A;
    A = A_org + 1e-5*randn(n,m);
    
    
    
    % Create the problem structure
    % n x m matrices with positive entries such that each column sums to 1.
    % Dimension is (n-1)*m.
    problem.M = multinomialfactory(n, m);
    
    % Cost
    problem.cost = @cost;
    function f = cost(X)
        f = 0.5*norm(X - A, 'fro')^2;
    end
    
    % Gradient
    problem.egrad = @egrad;
    function grad = egrad(X)
        grad = X - A; % Euclidean gradient
    end
    
    
    % Hessian
    problem.ehess = @ehess;
    function ehess = ehess(X, eta)
        ehess = eta;
    end
    
    
    % Check numerically whether gradient and Hessian are correct
    checkgradient(problem);
    drawnow;
    pause;
    checkhessian(problem);
    drawnow;
    pause;
    
    
    
    % Pick an algorithm to solve the problem
    [Xsol, costopt, infos] = trustregions(problem,[]);
    
end