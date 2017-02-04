function vec = tangent2vec(M, x, basis, u)
% Expands a tangent vector into an orthogonal basis in the Manopt framework
%
% vec = tangent2vec(M, x, basis, u)
%
% M is a Manopt manifold structure obtained from a factory.
% x is a point on the manifold M.
% basis is a cell containing n orthonormal tangent vectors at x, forming an
%       orthonormal basis of the tangent space at x.
% u is a tangent vector at x
%
% vec is a column vector of length n which contains the coefficients of the
%     expansion of u into the basis. Thus:
%
%    vec(k) = <basis{k}, u>_x          <- vec = tangent2vec(M, x, basis, u)
%
%    u = sum_{k=1}^n  vec(k)*basis{k}    <- u = lincomb(M, x, basis, vec)
%
% See also: tangentorthobasis lincomb orthogonalize grammatrix

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Feb. 3, 2017.
% Contributors: 
% Change log: 


    n = numel(basis);
    
    vec = zeros(n, 1);
    
    for k = 1 : n
        
        vec(k) = M.inner(x, basis{k}, u);
        
    end

end
