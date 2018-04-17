function Test_positivedefinite()
% function Test_positivedefinite()
% Test for sympositivedefinite geometry (matrix completion)
%

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, August 29, 2013.
% Contributors:
% Change log:

    % Problem
    n = 100;
    B = randn(n, n);
    C = B*B';
    
    
    % Create the manifold structure
    problem.M = sympositivedefinitefactory(n);
    
    
    % creating a symmetric matrix of ones and zeros
    f = 0.1; % % fraction of ones
    length_of_vector = n*(n-1)/2;
    test =(rand(length_of_vector,1) > f); % a vector with fraction f of zeros
    train = logical(ones(length_of_vector,1) - test); % a vector with fraction f of ones
    mask = sparse(squareform(train));
    
        
    % cost description
    problem.cost = @cost;
    function f = cost(X)
        f = .5*norm(mask.*(X - C), 'fro')^2;
    end
    
    
    % gradient description
    problem.grad = @(X) problem.M.egrad2rgrad(X, egrad(X));
    function g = egrad(X)
        g = mask.^2 .* (X - C);
    end
    
    
    % Hessian description
    problem.hess = @(X, U) problem.M.ehess2rhess(X, egrad(X), ehess(X, U), U);
    function Hess = ehess(X, eta)
        Hess = (mask.^2).*eta;
    end
    
    
    % Check numerically whether gradient and Ressian are correct
%     checkgradient(problem);
%     drawnow;
%     pause;
%     checkhessian(problem);
%     drawnow;
%     pause;
    
    
    % Initialization
    X0 = [];
    
    % Options (not mandatory)
    options.maxiter = 1000;
    options.maxinner = 30;
    options.maxtime = 120;
    options.tolgradnorm = 1e-16;
    
    % Pick an algorithm to solve the problem
    [Xopt, costopt, info] = trustregions(problem, X0, options);
    %     [Xopt costopt info] = conjugategradient(problem, X0, options);
    %     [Xopt costopt info] = steepestdescent(problem, X0, options);
    
end

