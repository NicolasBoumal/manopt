function [hessfd, storedb] = getHessianFD(problem, x, d, storedb)
% Computes an approx. of the Hessian w/ finite differences of the gradient.
%
% function [hessfd, storedb] = getHessianFD(problem, x, d, storedb)
%
% Return a finite difference approximation of the Hessian at x along d of
% the cost function described in the problem structure. The cache database
% storedb is passed along, possibly modified and returned in the process.
% The finite difference is based on computations of the gradient. 
%
% If the gradient cannot be computed, an exception is thrown.

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    % Import necessary tools etc. here
    import manopt.privatetools.*;
    
    if ~canGetGradient(problem)
        up = MException('manopt:getHessianFD:nogradient', ...
            'getHessianFD requires the gradient to be computable.');
        throw(up);
    end
    
    % First, check whether the step d is not too small
    if problem.M.norm(x, d) < eps
        hessfd = problem.M.zerovec(x);
        return;
    end
    
    % Parameter: how far do we look? TODO: give the user control over this.
    epsilon = 1e-4;
        
    % TODO: deal with the sign
    % sg = sign(d(find(d(:), 1, 'first')));
    sg = 1;
    norm_d = problem.M.norm(x, d);
    c = epsilon*sg/norm_d;
    
    % Gradient here
    [grad0 storedb] = getGradient(problem, x, storedb);
    
    % Point and gradient a little further along d
    x1 = problem.M.retr(x, d, c);
    [grad1 storedb] = getGradient(problem, x1, storedb);
    
    % Transport grad1 back from x1 to x.
    grad1 = problem.M.transp(x1, x, grad1);
    
    % Finite difference of them
    hessfd = problem.M.lincomb(x, 1/c, grad1, -1/c, grad0);
    
end
