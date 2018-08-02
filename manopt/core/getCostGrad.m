function [cost, grad] = getCostGrad(problem, x, storedb, key)
% Computes the cost function and the gradient at x in one call if possible.
%
% function [cost, grad] = getCostGrad(problem, x)
% function [cost, grad] = getCostGrad(problem, x, storedb)
% function [cost, grad] = getCostGrad(problem, x, storedb, key)
%
% Returns the value at x of the cost function described in the problem
% structure, as well as the gradient at x.
%
% storedb is a StoreDB object, key is the StoreDB key to point x.
%
% See also: canGetCost canGetGradient getCost getGradient

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%
%   Aug. 2, 2018 (NB):
%       The value of the cost function is now always cached.

    % Allow omission of the key, and even of storedb.
    if ~exist('key', 'var')
        if ~exist('storedb', 'var')
            storedb = StoreDB();
        end
        key = storedb.getNewKey();
    end
    
    
    % If the cost value was cached, read it and just compute the gradient.
    store = storedb.getWithShared(key);
    store_is_stale = false;
    
    if isfield(store, 'cost__')
        cost = store.cost__;
        grad = getGradient(problem, x, storedb, key);
        return;
    end


    if isfield(problem, 'costgrad')
    %% Compute the cost/grad pair using costgrad.
    
        % Check whether this function wants to deal with storedb or not.
        switch nargin(problem.costgrad)
            case 1
                [cost, grad] = problem.costgrad(x);
            case 2
                % Obtain, pass along, and save the store for x.
                [cost, grad, store] = problem.costgrad(x, store);
            case 3
                % Pass along the whole storedb (by reference), with key.
                [cost, grad] = problem.costgrad(x, storedb, key);
                store_is_stale = true;
            otherwise
                up = MException('manopt:getCostGrad:badcostgrad', ...
                    'costgrad should accept 1, 2 or 3 inputs.');
                throw(up);
        end

    else
    %% Revert to calling getCost and getGradient separately
    
        cost = getCost(problem, x, storedb, key);
        grad = getGradient(problem, x, storedb, key);
        store_is_stale = true;
        
    end
    
    if store_is_stale
        store = storedb.getWithShared(key);
    end
    
    % Cache the cost value.
    store.cost__ = cost;
    
    storedb.setWithShared(store, key);
    
end
