function orthobasis = orthogonalize(M, x, basis)
% Orthonormalizes a basis of tangent vectors in the Manopt framework.
%
% function orthobasis = orthogonalize(M, x, basis)
%
% M is a Manopt manifold structure obtained from a factory.
% x is a point on the manifold M.
% basis is a cell containing n linearly independent tangent vectors at x.
%
% orthobasis is a cell of same size as basis which contains an orthonormal
% basis for the same subspace as that spanned by basis. Orthonormality is
% assessed with respect to the metric on the tangent space to M at x.
%
% See also: grammatrix tangentorthobasis

% This file is part of Manopt: www.manopt.org. 
% Original author: Nicolas Boumal, April 28, 2016.
% Contributors: 
% Change log: 


    n = numel(basis);
    orthobasis = cell(size(basis));
    
    % Build the Gram matrix of the basis vectors.
    G = grammatrix(M, x, basis);
    
    % If the vectors in 'basis' were the columns of V, and the inner
    % product were the classical dot product, then G = V'*V. We are looking
    % for R, an invertible matrix such that V*R is orthogonal. Thus, R
    % satisfies R'*V'*V*R = eye(n); equivalently:
    %  G = inv(R)'*inv(R).
    % Computing a Cholesky factorization of G yields L such that G = L'*L.
    % Thus, R = inv(L). Each column of R states exactly which linear
    % combinations of the vectors in 'basis' must be computed to produce
    % the orthonormal basis.
    %
    % Of course, in that formalism, we could directly take a qr of V, but
    % in the actual setting V is not available; the only simple object
    % available is G.
	%
	% If it proves necessary, it may be good to try to avoid calling
	% the inv function on L explicitly.
    L = chol(G);
    R = inv(L);
    
    % Note that R is upper triangular.
    % We now compute the n linear combinations.
    
    for k = 1 : n
        
        orthobasis{k} = lincomb(M, x, basis(1:k), R(1:k, k));
        
    end

end
