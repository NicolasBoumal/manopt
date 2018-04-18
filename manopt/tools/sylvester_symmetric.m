function X = lyapunov_symmetric(A, C, tol)
% Solves AX + XA = C when A = A', as a pseudo-inverse.
%
% function X = lyapunov_symmetric(A, C)
% function X = lyapunov_symmetric(A, C, tol)
%
% When the solution exists and is unique, this is equivalent to
% sylvester(A, A', C) or lyap(A, -C). This works for both real and complex
% inputs.
% 
% If the solution is not unique, the smallest-norm solution is returned.
%
% If a solution does not exist, a minimum-residue solution is returned.
%
% tol is a tolerance used to determine which eigenvalyes are numerically
% zero. It can be specified manually; otherwise, a default value is chosen.
%
% Overall, if A is nxn, the output is very close to:
%   X = reshape(pinv(kron(A, eye(n)) + kron(eye(n), A))*C(:), [n, n]),
% but it is computed far more efficiently. The most expensive step is an
% eigendecomposition of A, whose complexity is essentially O(n^3) flops.
%
% If C is (skew-)symmetric, then X is (skew-)symmetric (up to round-off),
% and similarly in the complex case.

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, April 17, 2018.
% Contributors: 
% Change log: 

    % Make sure A is numerically Hermitian (or symmetric).
    % The cost of this safety step is negligible compared to eig.
    A = (A+A')/2;

    % V is unitary or orthogonal and lambda is real.
    [V, lambda] = eig(A, 'vector');
    
    % AX + XA = C  is equivalent to DY + YD = M with
    % Y = V'XV, M = V'CV and D = diag(lambda).
    M = V'*C*V;
    
    % W(i, j) = lambda(i) + lambda(j)
    W = bsxfun(@plus, lambda, lambda');
    
    % Normally, the solution Y is simply this:
    Y = M ./ W;
    
    % But this may involve divisions by (almost) 0 in certain places.
    % Thus, we go for a pseudo-inverse.
    absW = abs(W);
    if ~exist('tol', 'var') || isempty(tol)
        tol = numel(A)*eps(max(absW(:))); % similar to pinv tolerance
    end
    Y(absW <= tol) = 0;
    
    % Undo the change of variable
    X = V*Y*V';

end
