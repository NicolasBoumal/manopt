function [hess, storedb] = getHessian(problem, x, d, storedb)
% Computes the Hessian of the cost function at x along d.
%
% function [hess, storedb] = getHessian(problem, x, d, storedb)
%
% Returns the Hessian at x along d of the cost function described in the
% problem structure. The cache database storedb is passed along, possibly
% modified and returned in the process.
%
% If an exact Hessian is not provided, an approximate Hessian is returned
% if possible, without warning. If not possible, an exception will be
% thrown. To check whether an exact Hessian is available or not (typically
% to issue a warning if not), use canGetHessian.
%
% See also: getPrecon getApproxHessian canGetHessian

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    % Import necessary tools etc. here
    import manopt.privatetools.*;
    
    if isfield(problem, 'hess')
    %% Compute the Hessian using hess.
        
        % Check whether the Hessian function wants to deal with the store
        % structure or not.
        switch nargin(problem.hess)
            case 2
                hess = problem.hess(x, d);
            case 3
                % Obtain, pass along, and save the store structure
                % associated to this point.
                store = getStore(problem, x, storedb);
                [hess store] = problem.hess(x, d, store);
                storedb = setStore(problem, x, storedb, store);
            otherwise
                up = MException('manopt:getHessian:badhess', ...
                    'hess should accept 2 or 3 inputs.');
                throw(up);
        end
        
    else
    %% Attempt the computation of an approximation of the Hessian.
        
        [hess, storedb] = getApproxHessian(problem, x, d, storedb);
        
    end
    
end
