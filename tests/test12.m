function [] = test12()
% function test12()
%
% Test for Nelder Mead
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
    
    % Pick the manifold
    n = 5;
    problem.M = euclideanfactory(n);

    problem.cost = @rosen;
%     problem.cost = @quadratic;
    
    % Define Rosenbrock function
    function y = rosen(x)
        % Rosenbrock function
        % Matlab Code by A. Hedar (Nov. 23, 2005).
        sum = 0;
        for j = 1:n-1;
            sum = sum+100*(x(j)^2-x(j+1))^2+(x(j)-1)^2;
        end
        y = sum;
    end

    % Super simple function
    function y = quadratic(x)
        y = x'*x;
    end

    options.maxcostevals = 1000;
    [x bestcost info] = neldermead(problem, [], options);
    
    keyboard;

end
