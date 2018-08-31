function grad = getGradient(problem, x, storedb, key)
% Computes the gradient of the cost function at x.
%
% function grad = getGradient(problem, x)
% function grad = getGradient(problem, x, storedb)
% function grad = getGradient(problem, x, storedb, key)
%
% Returns the gradient at x of the cost function described in the problem
% structure.
%
% storedb is a StoreDB object, key is the StoreDB key to point x.
%
% See also: getDirectionalDerivative canGetGradient

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%
%  June 28, 2016 (NB):
%       Works with getPartialGradient.
%
%   Nov. 1, 2016 (NB):
%       Added support for gradient from directional derivatives.
%       Last resort is call to getApproxGradient instead of an exception.

    % Allow omission of the key, and even of storedb.
    if ~exist('key', 'var')
        if ~exist('storedb', 'var')
            storedb = StoreDB();
        end
        key = storedb.getNewKey();
    end

    
    if isfield(problem, 'grad')
    %% Compute the gradient using grad.
	
        % Check whether this function wants to deal with storedb or not.
        switch nargin(problem.grad)
            case 1
                grad = problem.grad(x);
            case 2
                % Obtain, pass along, and save the store for x.
                store = storedb.getWithShared(key);
                [grad, store] = problem.grad(x, store);
                storedb.setWithShared(store, key);
            case 3
                % Pass along the whole storedb (by reference), with key.
                grad = problem.grad(x, storedb, key);
            otherwise
                up = MException('manopt:getGradient:badgrad', ...
                    'grad should accept 1, 2 or 3 inputs.');
                throw(up);
        end
    
    elseif isfield(problem, 'costgrad')
    %% Compute the gradient using costgrad.
		
        % Note: here, we could in principle cache the value of the cost
        % function. We do not, as it is unlikely that an algorithm which
        % requires the cost value would query the gradient before it
        % queries the gradient; thus, in all likelihood, the cost value has
        % already been computed (and cached) at the current point.
    
        % Check whether this function wants to deal with storedb or not.
        switch nargin(problem.costgrad)
            case 1
                [unused, grad] = problem.costgrad(x); %#ok
            case 2
                % Obtain, pass along, and save the store for x.
                store = storedb.getWithShared(key);
                [unused, grad, store] = problem.costgrad(x, store); %#ok
                storedb.setWithShared(store, key);
            case 3
                % Pass along the whole storedb (by reference), with key.
                [unused, grad] = problem.costgrad(x, storedb, key); %#ok
            otherwise
                up = MException('manopt:getGradient:badcostgrad', ...
                    'costgrad should accept 1, 2 or 3 inputs.');
                throw(up);
        end
    
    elseif canGetEuclideanGradient(problem)
    %% Compute the gradient using the Euclidean gradient.
        
        egrad = getEuclideanGradient(problem, x, storedb, key);
        grad = problem.M.egrad2rgrad(x, egrad);
    
    elseif canGetPartialGradient(problem)
    %% Compute the gradient using a full partial gradient.
        
        d = problem.ncostterms;
        grad = getPartialGradient(problem, x, 1:d, storedb, key);
        
    elseif canGetDirectionalDerivative(problem)
    %% Compute gradient based on directional derivatives; expensive!
    
        B = tangentorthobasis(problem.M, x);
        df = zeros(size(B));
        for k = 1 : numel(B)
            df(k) = getDirectionalDerivative(problem, x, B{k}, storedb, key);
        end
        grad = lincomb(problem.M, x, B, df);

    else
    %% Attempt the computation of an approximation of the gradient.
        
        grad = getApproxGradient(problem, x, storedb, key);
        
    end
    
end
