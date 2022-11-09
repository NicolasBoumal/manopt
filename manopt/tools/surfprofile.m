function [costs, x, d1, d2, t1, t2] = surfprofile(problem, x, d1, d2, t1, t2)
% Plot the cost function as a surface over a 2-dimensional subspace.
%
% function surfprofile(problem)
% function surfprofile(problem, x)
% function surfprofile(problem, x, d1)
% function surfprofile(problem, x, d1, d2)
% function surfprofile(problem, x, [], [], t1, t2)
% function surfprofile(problem, x, d1, d2, t1, t2)
% function [costs, x, d1, d2, t1, t2] = surfprofile(...)
%
% Evaluates the cost function at points
%
%   gamma(t1, t2) = exponential_x(t1*d1 + t2*d2)
% 
% where the exponential map at x is specified by problem.M.exp.
% If M.exp is not available, then M.retr is used instead.
% Vectors d1 and d2 are two tangent vectors to M at the point x.
% The values assigned to t1 and t2 are as specified in the two input
% vectors t1 and t2.
% 
% If the function is called with an output, the plot is not drawn and the
% cost values are returned in a matrix of size length(t1)*length(t2).
% To plot as a surface, call surf(t1, t2, costs.'); (notice the transpose)
%
% If x is omitted, a point is generated at random.
% If d1 is omitted, a random unit-norm tangent vector at x is generated.
% If d2 is omitted, a random unit-norm tangent vector at x is generated,
% orthogonally to d1.
% If t1 or t2 are omitted, they are generated with linspace's in [-1, 1].
%
% See also plotprofile

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Sep. 1, 2014.
% Contributors: 
% Change log: 
%
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%
%   Nov. 12, 2016 (NB):
%       Most inputs are now optional.
%
%   Nov.  9, 2022 (NB):
%       When generated randomly, d1, d2 were not necessarily unit-norm.
%       Now they are. Also, x, d1, d2, t1, t2 are returned as outputs.

    % Verify that the problem description is sufficient.
    if ~canGetCost(problem)
        error('It seems no cost function was provided.');  
    end

    M = problem.M;

    if ~exist('x', 'var') || isempty(x)
        x = M.rand();
        if (exist('d1', 'var') && ~isempty(d1)) || ...
           (exist('d2', 'var') && ~isempty(d2))
            error('If x is omitted, d1, d2 should not be specified.');
        end
    end
    if ~exist('d1', 'var') || isempty(d1)
        d1 = M.randvec(x);
        d1 = M.lincomb(x, 1/M.norm(x, d1), d1);
        if (exist('d2', 'var') && ~isempty(d2))
            error('If d1 is omitted, d2 should not be specified.');
        end
    end
    if ~exist('d2', 'var') || isempty(d2)
        d2 = M.randvec(x);
        % Make d2 orthonormal to d1.
        % Note: d1 may have been provided by the user, so we cannot assume
        % that it has unit norm.
        coeff = M.inner(x, d1, d2) / M.inner(x, d1, d1);
        d2 = M.lincomb(x, 1, d2, -coeff, d1);
        d2 = M.lincomb(x, 1/M.norm(x, d2), d2);
        % If both d1, d2 have been generated at random, then they are
        % orthonormal. To check, verify the following is numerically zero:
        %   grammatrix(M, x, {d1, d2}) - eye(2)
    end

    if ~exist('t1', 'var') || isempty(t1)
        t1 = linspace(-1, 1, 51);
    end
    if ~exist('t2', 'var') || isempty(t2)
        t2 = linspace(-1, 1, 51);
    end
    
    if isfield(M, 'exp')
        expo = M.exp;
        str = 'Exp';
    else
        expo = M.retr;
        str = 'Retr';
    end
    
    storedb = StoreDB();
    linesearch_fun = @(ta, tb) getCost(problem, ...
                         expo(x, M.lincomb(x, ta, d1, tb, d2)), ...
                         storedb);
    
    costs = zeros(length(t1), length(t2));
    for i = 1 : length(t1)
        for j = 1 : length(t2)
            costs(i, j) = linesearch_fun(t1(i), t2(j));
        end
    end
    
    if nargout == 0
        surf(t1, t2, costs.');
        xlabel('t1');
        ylabel('t2');
        zlabel(['f(' str '_x(t1*d1+t2*d2))']);
    end
    
end
