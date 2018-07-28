function egrad = getEuclideanGradient(problem, x, storedb, key)
% Computes the Euclidean gradient of the cost function at x.
%
% function egrad = getEuclideanGradient(problem, x)
% function egrad = getEuclideanGradient(problem, x, storedb)
% function egrad = getEuclideanGradient(problem, x, storedb, key)
%
% Returns the Euclidean gradient at x of the cost function described in the
% problem structure.
%
% storedb is a StoreDB object, key is the StoreDB key to point x.
%
% Because computing the Hessian based on the Euclidean Hessian will require
% the Euclidean gradient every time, to avoid overly redundant
% computations, if the egrad function does not use the store caching
% capabilites, this implements an automatic caching functionality. Writing
% egrad to accept the optional store or storedb parameter will disable
% automatic caching, but allow user controlled caching.
%
% See also: getGradient canGetGradient canGetEuclideanGradient

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 9, 2013.
% Contributors: 
% Change log: 
%
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%
%   June 28, 2016 (NB):
%       Added support for getPartialEuclideanGradient
%
%   July 26, 2018 (NB):
%       The Euclidean gradient is now automatically cached if the Euclidean
%       Hessian is also computable from the problem description. This
%       differs from previous behavior where it would only be cached if
%       problem.egrad did not accept store or storedb as input; the
%       converse was taken as a sign that the user wants to deal with
%       caching on their own, but in reality it proved more confusing than
%       helpful.

    % Allow omission of the key, and even of storedb.
    if ~exist('key', 'var')
        if ~exist('storedb', 'var')
            storedb = StoreDB();
        end
        key = storedb.getNewKey();
    end

    % Contrary to most similar functions, here, we get the store by
    % default. This is for the special caching functionality described
    % below.
    store = storedb.getWithShared(key);

    % If the Euclidean Hessian can be computed from the problem
    % definition, it is likely that the user will use it. To get the
    % Riemannian Hessian from the Euclidean Hessian usually requires
    % the Euclidean gradient. Since there is a significant cost
    % associated to computing the Euclidean gradient, conservatively,
    % we force caching of the Euclidean gradient in that scenario.
    force_caching = canGetEuclideanHessian(problem);
    if force_caching && isfield(store, 'egrad__')
        egrad = store.egrad__;
        return;
    end
    
    if isfield(problem, 'egrad')
    %% Compute the Euclidean gradient using egrad.
	
        % Check whether this function wants to deal with storedb or not.
        switch nargin(problem.egrad)
            case 1
                egrad = problem.egrad(x);
            case 2
                [egrad, store] = problem.egrad(x, store);
            case 3
                egrad = problem.egrad(x, storedb, key);
            otherwise
                up = MException('manopt:getEuclideanGradient:badegrad', ...
                    'egrad should accept 1, 2 or 3 inputs.');
                throw(up);
        end
        
    elseif canGetPartialEuclideanGradient(problem)
    %% Compute the Euclidean gradient using a full partial Euclidean gradient.
        
        d = problem.ncostterms;
        egrad = getPartialEuclideanGradient(problem, x, 1:d, storedb, key);

    else
    %% Abandon computing the Euclidean gradient
    
        up = MException('manopt:getEuclideanGradient:fail', ...
            ['The problem description is not explicit enough to ' ...
             'compute the Euclidean gradient of the cost.']);
        throw(up);
        
    end
    
    % Cache here.
    if force_caching
        store.egrad__ = egrad;
    end

    storedb.setWithShared(store, key);
    
end
