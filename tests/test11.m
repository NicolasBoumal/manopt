function [] = test11()
% function test11()
%
% Try out centroid function in Nelder-Mead solver.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
    
    % Pick the manifold
    n = 10;
    k = 2;
%     M = euclideanfactory(n);
    M = obliquefactory(n, k);

    % Generate a bunch of points
    m = 25;
    x = cell(m, 1);
    for i = 1 : m
        x{i} = M.rand();
    end
    
    y = centroid(M, x);
    
    % This is for Euclidean only
    if strfind(M.name(), 'Euclidean')
        xx = zeros(n, 1);
        for i = 1 : m
            xx = xx + x{i}/m;
        end
        fprintf('Distance between mean point and centroid found: %e.\n',...
                 norm(xx-y));
    end
    
end
