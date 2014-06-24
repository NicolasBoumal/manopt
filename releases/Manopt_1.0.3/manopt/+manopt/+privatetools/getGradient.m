function [grad, storedb] = getGradient(problem, x, storedb)
% Computes the gradient of the cost function at x.
%
% function [grad, storedb] = getGradient(problem, x, storedb)
%
% Returns the gradient at x of the cost function described in the problem
% structure. The cache database storedb is passed along, possibly modified
% and returned in the process.
%
% See also: getDirectionalDerivative canGetGradient

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    % Import necessary tools etc. here
    import manopt.privatetools.*;
    
    if isfield(problem, 'grad')
    %% Compute the gradient using grad.
        
        % Check whether the gradient function wants to deal with the store
        % structure or not
        switch nargin(problem.grad)
            case 1
                grad = problem.grad(x);
            case 2
                % Obtain, pass along, and save the store structure
                % associated to this point.
                store = getStore(problem, x, storedb);
                [grad store] = problem.grad(x, store);
                storedb = setStore(problem, x, storedb, store);
            otherwise
                up = MException('manopt:getGradient:badgrad', ...
                    'grad should accept 1 or 2 inputs.');
                throw(up);
        end
    
    elseif isfield(problem, 'costgrad')
    %% Compute the gradient using costgrad.
        
        % Check whether the costgrad function wants to deal with the store
        % structure or not
        switch nargin(problem.costgrad)
            case 1
                [unused, grad] = problem.costgrad(x); %#ok
            case 2
                % Obtain, pass along, and save the store structure
                % associated to this point.
                store = getStore(problem, x, storedb);
                [unused, grad, store] = problem.costgrad(x, store); %#ok
                storedb = setStore(problem, x, storedb, store);
            otherwise
                up = MException('manopt:getGradient:badcostgrad', ...
                    'costgrad should accept 1 or 2 inputs.');
                throw(up);
        end
    
    elseif canGetEuclideanGradient(problem)
    %% Compute the gradient using the Euclidean gradient.
        
        [egrad, storedb] = getEuclideanGradient(problem, x, storedb);
        grad = problem.M.egrad2rgrad(x, egrad);

    else
    %% Abandon computing the gradient
    
        up = MException('manopt:getGradient:fail', ...
            ['The problem description is not explicit enough to ' ...
             'compute the gradient of the cost.']);
        throw(up);
        
    end
    
end
