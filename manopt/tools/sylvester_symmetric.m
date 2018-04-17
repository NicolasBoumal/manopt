function X = sylvester_symmetric(A, C, tol)
% Solves AX + XA = C when A = A' and C are real.
%
% function X = sylvester_symmetric(A, C)
% function X = sylvester_symmetric(A, C, tol)
%
% When the solution exists and is unique, this is equivalent to
% sylvester(A, A', C) (though it may differ numerically.)
% 
% If the solution is not unique, the smallest norm solution is returned.
% If C is symmetric, then X is always symmetric.
%
% If a solution does not exist, a minimum-residue solution is returned.
%
% tol is a tolerance used to determine which entries are numerically zero
% in an intermediate weight matrix. It can be specified manually;
% otherwise, a default value is chosen.
%
% The complexity is ~O(n^3) if A, C are nxn.

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, April 17, 2018.
% Contributors: 
% Change log: 

    assert(isreal(A) && isreal(C), 'Code for real inputs only.');

    % Make sure A is numerically symmetric (the cost of this safety step is
    % negligible compared to the call to eig.)
    A = (A+A')/2;

    % V is orthogonal and lambda is real.
    [V, lambda] = eig(A, 'vector');
    
    % AX + XA = C  ===  DY + YD = M with Y = V'XV, M = V'CV and D = diag(lambda)
    M = V'*C*V;
    
    % W(i, j) = lambda(i) + lambda(j)
    W = bsxfun(@plus, lambda, lambda');
    
    % Normally, the solution Y is simply the following:
    Y = M ./ W;
    
    % But this may involve divisions by (almost) 0 in certain places.
    % Assuming a solution exists, if W(i, j) is numerically zero,
    % then M(i, j) should also be numerically zero. In that scenario, the
    % smallest norm solution will have Y(i, j) = 0 (since the Frobenius
    % norms of X and Y are the same.)
    if ~exist('tol', 'var') || isempty(tol)
        tol = eps(max(abs(lambda)));
    end
    mask = (W <= tol);
    Y(mask) = 0;
    
    X = V*Y*V';

end
