function X = solve_for_triu(A, H)
% Solve the linear matrix equation AX + X'A' = H for X upper triangular.
%
% function X = solve_for_triu(A, H)
%
% Given A of size p-by-p and H (symmetric) of size p-by-p, solves the
% linear matrix equation AX + X'A' = H for X upper triangular.
% 
% The total computational cost is O(p^4).
%
% If the same equation is to be solved for X symmetric instead, call
% Matlab's built-in sylvester function.
%
% This is a support function to compute the inverse of QR-based
% retractions.
%
% See also: stiefelfactory rotationsfactory sylvester

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 18, 2018.
% Contributors: 
% Change log:

% One tentative idea to reduce the cost to O(p^3) involves taking an LU
% factorization of A. Then, we obtain a permutation matrix P and
% triangular matrices L (lower) and U (upper) such that PA = LU.
% Since inv(P) = P', the matrix equation becomes:
%
%  P' L U X + X' U' L' P = H
%
% Notice that U*X is still upper triangular, so that we can solve for
% U*X first, and obtain X later by solving an upper triangular system.
% After this change of variables, the system involves P'L instead of A.
% If the permutation happens to be identity, then clearly principal
% submatrices of P'L = L are lower triangular themselves, and as a
% result the linear systems that we need to solve below only cost
% O(pp^2) instead of O(pp^3). Summing for pp = 1 .. p gives O(p^3)
% instead of O(p^4). In general though, P is not the identity
% permutation, and it would require extra work to exploit the
% particular structure of the principal submatrices of P'L (if at all
% possible.) We do not pursue this strategy below.

    p = size(A, 1);
    X = zeros(p, p);
    for pp = 1 : p
        b = H(1:pp, pp);
        b(end) = b(end)/2;
        b(1:end-1) = b(1:end-1) - X(1:pp-1, 1:pp-1)'*A(pp, 1:pp-1)';
        X(1:pp, pp) = A(1:pp, 1:pp) \ b;
    end
    
end
