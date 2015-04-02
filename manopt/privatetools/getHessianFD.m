function [hessfd, storedb] = getHessianFD(problem, x, d, storedb)
% Computes an approx. of the Hessian w/ finite differences of the gradient.
%
% function [hessfd, storedb] = getHessianFD(problem, x, d, storedb)
%
% Returns a finite difference approximation of the Hessian at x along d of
% the cost function described in the problem structure. The cache database
% storedb is passed along, possibly modified and returned in the process.
% The finite difference is based on computations of the gradient. 
%
% If the gradient cannot be computed, an exception is thrown.

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   Feb. 19, 2015 (NB):
%       It is sufficient to ensure positive radial linearity to guarantee
%       (together with other assumptions) that this approximation of the
%       Hessian will confer global convergence to the trust-regions method.
%       Formerly, in-code comments referred to the necessity of having
%       complete radial linearity, and that this was harder to achieve.
%       This appears not to be necessary after all, which simplifies the
%       code.

    
    if ~canGetGradient(problem)
        up = MException('manopt:getHessianFD:nogradient', ...
            'getHessianFD requires the gradient to be computable.');
        throw(up);
    end
	
	% Step size
    norm_d = problem.M.norm(x, d);
    
    % First, check whether the step d is not too small
    if norm_d < eps
        hessfd = problem.M.zerovec(x);
        return;
    end
    
    % Parameter: how far do we look?
	% TODO: this parameter should be tunable by the user.
    epsilon = 1e-4;
        
    c = epsilon/norm_d;
    
    % Compute the gradient at the current point.
    [grad0, storedb] = getGradient(problem, x, storedb);
    
    % Compute a point a little further along d and the gradient there.
    x1 = problem.M.retr(x, d, c);
    [grad1, storedb] = getGradient(problem, x1, storedb);
    
    % Transport grad1 back from x1 to x.
    grad1 = problem.M.transp(x1, x, grad1);
    
    % Return the finite difference of them.
    hessfd = problem.M.lincomb(x, 1/c, grad1, -1/c, grad0);
    
end
