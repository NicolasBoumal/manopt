function [Q, R] = stiefelgeneralized_retraction_MGS(B, X, U, t)
% Retraction for generalized Stiefel based on Modified Gram-Schmidt.
% When used just as a retraction, only the output Q is relevant.
% NB, Dec. 16, 2018.
    if ~exist('t', 'var') || isempty(t)
        A = X + U;   % t = 1 by default
    else
        A = X + t*U;
    end
    [n, p] = size(X);
    Q = zeros(n, p);
    R = zeros(p, p);
    for j = 1 : p
        v = A(:, j);
        R(j, j) = sqrt(v'*B*v);
        Q(:, j) = v / R(j, j);
        R(j, (j+1):p) = Q(:, j)' * B * A(:, (j+1):p);
        A(:, (j+1):p) = A(:, (j+1):p) - Q(:, j) * R(j, (j+1):p);
    end
end
