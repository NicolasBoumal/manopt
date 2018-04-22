function B = doubly_stochastic(A, maxiter, mode)
% Project a matrix to the doubly stochastic matrices (Sinkhorn's algorithm)
%
% function B = doubly_stochastic(A)
% function B = doubly_stochastic(A, maxiter)
% function B = doubly_stochastic(A, [], mode)
% function B = doubly_stochastic(A, maxiter, mode)
%
% Given an element-wise non-negative matrix A of size nxn, returns a
% doubly-stochastic matrix B of size nxn by applying Sinkhorn's algorithm
% to A.
% 
% maxiter (optional): strictly positive integer representing the maximum 
%	number of iterations of Sinkhorn's algorithm. 
%	The default value of maxiter is n^2.
% mode (optional): Setting mode = 1 changes the behavior of the algorithm 
% 	such that the input A is an n x p matrix with AA' having 
%	element-wise non-negative entries and the output B is also n x p
%	such that BB' is a doubly-stochastic matrix. The default value is 0.

% The file is based on developments in the research paper
% Philip A. Knight, "The Sinkhorn–Knopp Algorithm: Convergence and 
% Applications" in SIAM Journal on Matrix Analysis and Applications 30(1), 
% 261-275, 2008.
%
% Please cite the Manopt paper as well as the research paper.

% This file is part of Manopt: www.manopt.org.
% Original author: David Young, September 10, 2015.
% Contributors: Ahmed Douik, March 15, 2018.
% Change log:

    n = size(A, 1);
    tol = eps(n);
    
    if ~exist('maxiter', 'var') || isempty(maxiter)
        maxiter = n^2;
    end
    
    if ~exist('mode', 'var') || isempty(mode)
        mode = 0;
    end
    
    if mode == 1
        C = A*A';
    else
        C = A;
    end
    
    iter = 1;
    d_1 = 1./sum(C);
    d_2 = 1./(C * d_1.');
    while iter < maxiter
        iter = iter + 1;
        row = d_2.' * C;
        if  max(abs(row .* d_1 - 1)) <= tol
            break;
        end
        d_1 = 1./row;
        d_2 = 1./(C * d_1.');
    end
    
    if mode == 1
        v = sqrt(d_2(1)/d_1(1))*d_1;
        B = diag(v)*A;
    else
        B = C .* (d_2 * d_1);
    end
         
end
