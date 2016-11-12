function [H, basis] = hessianmatrix(problem, x, basis)
% Computes a matrix which represents the Hessian in some tangent basis.
%
% [H, basis] = hessianmatrix(problem, x)
% [H, basis] = hessianmatrix(problem, x, basis)
%
% problem is a Manopt problem structure with a manifold and cost function.
% x is a point on the manifold problem.M.
% basis (optional) is an orthonormal basis for the tangent space to the
% manifold at x. If no basis is supplied, one will be generated at random.
% If the basis spans only a subspace of the tangent space at x,
% then the returned matrix represents the Hessian restricted to that subspace.
%
% H is an n-by-n symmetric matrix (with n the number of vectors in the basis)
% such that H(i, j) is the inner product between basis{i}
% and Hess(basis{j}), with respect to the metric on the tangent space to
% problem.M at x, where Hess(basis{j}) is the vector obtained after
% applying the Hessian at x to basis{j}.
%
% For optimization, it is usually not useful to compute the Hessian matrix,
% as this quickly becomes expensive. This tool is provided mostly for
% exploration and debugging rather than to be used algorithmically in
% solvers. To access the spectrum of the Hessian, it may be more practical
% to call hessianextreme or hessianspectrum. This should coincide with eig(H).
%
% See also: hessianspectrum hessianextreme tangentorthobasis orthogonalize

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 14, 2016.
% Contributors: 
% Change log: 


    % Unless an orthonormal basis for the tangent space at x is provided,
    % pick a random one.
    if ~exist('basis', 'var') || isempty(basis)
	    n = problem.M.dim();
        basis = tangentorthobasis(problem.M, x, n);
	else
	    n = numel(basis);
    end
    
    % Create a store database and get a key for x
    storedb = StoreDB(1);
    key = storedb.getNewKey();
    
    % Apply the Hessian at x to each basis vector
    Hbasis = cell(n, 1);
    for k = 1 : numel(Hbasis)
        Hbasis{k} = getHessian(problem, x, basis{k}, storedb, key);
    end
    
    % H is the matrix which contains the inner products of
    % the ((basis vectors)) with the ((Hessian applied to basis vectors)).
    H = zeros(n);
    for i = 1 : n
        H(i, i) = problem.M.inner(x, basis{i}, Hbasis{i});
        for j = (i+1) : n
            H(i, j) = problem.M.inner(x, basis{i}, Hbasis{j});
            H(j, i) = H(i, j);
        end
    end

end
