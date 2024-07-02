function checkgradient(problem, x, d)
% Checks the consistency of the cost function and the gradient.
%
% function checkgradient(problem)
% function checkgradient(problem, x)
% function checkgradient(problem, x, d)
%
% checkgradient performs a numerical test to check that the gradient
% defined in the problem structure agrees up to first order with the cost
% function at some point x, along some direction d. The test is based on a
% truncated Taylor series (see online Manopt documentation).
%
% It is also tested that the gradient is indeed a tangent vector.
% 
% Both x and d are optional and will be sampled at random if omitted.
%
% See also: checkdiff checkhessian

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%
%   Nov. 1, 2016 (NB):
%       Now calls checkdiff with force_gradient = true, instead of doing an
%       rmfield of problem.diff. This became necessary after getGradient
%       was updated to know how to compute the gradient from directional
%       derivatives.
%
%   July 2, 2024 (NB):
%       Now using the new tool offtangent to check that the gradient is
%       a tangent vector. Together with improvements in hyperbolicfactory,
%       this notably made it possible to remove a message that was useful
%       only to a few users, and likely puzzling to all others. Some other
%       improvements to printed messages. Also styled the output with bold.

    
    fprintf('<strong># Gradient check</strong>\n');

    % Verify that the problem description is sufficient.
    if ~canGetCost(problem)
        % The call to canGetPartialGradient will readily issue a warning if
        % problem.ncostterms is not defined even though it is expected.
        if ~canGetPartialGradient(problem)
            error('getCost:checkgradient', 'It seems no cost was provided.');
        else
            error('getCost:stochastic', ['It seems no cost was provided.\n' ...
                  'If you intend to use a stochastic solver, you still\n' ...
                  'need to define problem.cost to use checkgradient.']);
        end
    end
    if ~canGetGradient(problem)
        warning('manopt:checkgradient:nograd', ...
                'It seems no gradient was provided.');
    end
        
    x_isprovided = exist('x', 'var') && ~isempty(x);
    d_isprovided = exist('d', 'var') && ~isempty(d);
    
    if ~x_isprovided && d_isprovided
        error('If d is provided, x must be too, since d is tangent at x.');
    end
    
    % If x and / or d are not specified, pick them at random.
    if ~x_isprovided
        x = problem.M.rand();
    end
    if ~d_isprovided
        d = problem.M.randvec(x);
    end

    %% Check that the gradient yields a first order model of the cost.
    
    % Call checkdiff, forcing it to use the gradient.
    force_gradient = true;
    checkdiff(problem, x, d, force_gradient);
    
    %% Try to check that the gradient is a tangent vector.
    storedb = StoreDB();
    key = storedb.getNewKey();
    grad = getGradient(problem, x, storedb, key);
    err = offtangent(problem.M, x, grad);
    if ~isnan(err)
        fprintf('The gradient at x must be a tangent vector at x.\n');
        fprintf(['If so, the following number is zero up to machine ' ...
                 'precision: <strong>%g</strong>.\n'], err);
        if ~isinf(err)
            fprintf('If it is far from 0, the gradient is not tangent.\n');
        else
            fprintf(['The output is Inf, suggesting the gradient is ' ...
                     'not in the right format.\nCheck array sizes.']);
        end
    else
        fprintf(['Unfortunately, Manopt was unable to verify that the '...
                 'gradient is indeed a tangent vector.\nPlease verify ' ...
                 'this manually or implement the ''tangent'' or the ' ...
                 '''offtangent'' function in your manifold structure.']);
    end

end
