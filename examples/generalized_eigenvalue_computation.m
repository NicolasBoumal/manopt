function [X, info] = generalized_eigenvalue_computation(A, B, p)
% Returns orthonormal basis of the dominant invariant p-subspace of B^-1 A.
%
% function X = dgeneralized_eigenvalue_computation(A, B, p)
%
% Input: A real, symmetric matrix A of size nxn and an integer p < n.
%        B is symmetric positive definite matrix.
% Output: A real, orthonormal matrix X of size nxp such that trace(X'*A*X)
%         is maximized subject to X'*B*X is identity. 
%         That is, the columns of X form an orthonormal basis
%         of a dominant subspace of dimension p of B^(-1)*A. These are thus
%         eigenvectors associated with the largest eigenvalues of B^(-1)*A 
%         (in no particular order). Sign is important: 2 is deemed a larger
%         eigenvalue than -5.
% 
% We intend to solve the system AX = BX Lambda.
%
%
% The optimization is performed on the generalized Grassmann manifold, 
% since only the space spanned by the columns of X matters. 
%
% The optimization problem that we are solving here is 
% maximize trace(X'*A*X) subject to X'*B*X = I. 
% Consequently, the solutions remain invariant to transformation
% X --> XO, where O is a p-by-p orthogonal matrix. The search space, in
% essence, is set of equivalence classes
% [X] = {XO : X'*B*X = I and O is orthogonal matrix}. This space is called
% the generalized Grassmann manifold.


% This file is part of Manopt and is copyrighted. See the license file.
%
% Main author: Bamdev Mishra, June 30, 2015.
% Contributors:
%
% Change log:
%
    
    % Generate some random data to test the function
    if ~exist('A', 'var') || isempty(A)
        A = randn(128);
        A = (A+A')/2;
    end
    if ~exist('B', 'var') || isempty(B)
        n = size(A, 1);
        e = ones(n, 1);
        B = spdiags([-e 2*e -e], -1:1, n, n); % Symmetric positive definite
    end
    
    if ~exist('p', 'var') || isempty(p)
        p = 3;
    end
    
    % Make sure the input matrix is square and symmetric
    n = size(A, 1);
	assert(isreal(A), 'A must be real.')
    assert(size(A, 2) == n, 'A must be square.');
    assert(norm(A-A', 'fro') < n*eps, 'A must be symmetric.');
	assert(p<=n, 'p must be smaller than n.');
    
    % Define the cost and its derivatives on the generalized 
    % Grassmann manifold, i.e., the column space of all X such that
    % X'*B*X is identity.  
    gGr = grassmanngeneralizedfactory(n, p, B);
    
    problem.M = gGr;
    problem.cost = @(X)    -trace(X'*A*X);
    problem.grad = @(X)    -2*gGr.egrad2rgrad(X, A*X);
    problem.hess = @(X, H) -2*gGr.ehess2rhess(X, A*X, A*H, H);
    
    % Execute some checks on the derivatives for early debugging.
    % These things can be commented out of course.
    % checkgradient(problem);
    % pause;
    % checkhessian(problem);
    % pause;
    
    % Issue a call to a solver. A random initial guess will be chosen and
    % default options are selected except for the ones we specify here.
    options.Delta_bar = 8*sqrt(p);
    [X, costX, info, options] = trustregions(problem, [], options); %#ok<ASGLU>
    
end
