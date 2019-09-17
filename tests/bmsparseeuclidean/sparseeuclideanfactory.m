function M = euclideansparsefactory(A)
% Returns a manifold struct to optimize over real matrices with given sparsity pattern.
%
% function M = euclideansparsefactory(A)
%
% Returns M, a structure describing the Euclidean space of real matrices,
% equipped with the standard Frobenius distance and associated trace inner
% product, as a manifold for Manopt.
%
% The matrices are represented as sparse matrices, that their sparsity pattern 
% is fixed. The tangent vectors are represented in the same way as points 
% since this is a Euclidean space.
%
% See also: euclideanfactory euclideancomplexfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Mar. 28, 2019.
% Change log: 
%    May 3, 2019 (NB): adapted many functions to take better advantage of sparsity.
	
    dimensions_vec = size(A);
    assert(length(dimensions_vec) == 2, 'A should be a matrix (or a vector).');
    [I, J] = find(A);
    nvals = length(I);
    S = sparse(I, J, ones(nvals, 1), dimensions_vec(1), dimensions_vec(2), nvals);
      
    M.size = @() dimensions_vec;
    
    M.name = @() sprintf('Euclidean space R^(%dx%d) with fixed sparsity pattern containg %d non-zero entries', ...
                                        dimensions_vec(1), dimensions_vec(2), nvals);
    
    M.dim = @() nvals;
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:);
    
    M.norm = @(x, d) norm(d, 'fro');
    
    M.dist = @(x, y) norm(x - y, 'fro');
    
    M.typicaldist = @() sqrt(prod(dimensions_vec));
    
    M.proj = @(x, d) S.*d;
    
    M.egrad2rgrad = @(x, g) S.*g;
    
    M.ehess2rhess = @(x, eg, eh, d) S.*eh;
    
    M.tangent = M.proj;
    
    M.exp = @exp;
    function y = exp(x, d, t)
        if nargin == 3
            y = x + t*d;
        else
            y = x + d;
        end
    end
    
    M.retr = M.exp;
	
    M.log = @(x, y) y-x;

    M.hash = @(x) ['z' hashmd5(nonzeros(x))];
    
    M.rand = @() sprandn(S);
    
    M.randvec = @randvec;
    function u = randvec(x) %#ok<INUSD>
        u = S.*randn(dimensions_vec);
        u = u / norm(u(:), 'fro');
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(x) zeros(dimensions_vec);
    
    M.transp = @(x1, x2, d) d;
    M.isotransp = M.transp; % the transport is isometric
    
    M.pairmean = @(x1, x2) .5*(x1+x2);
    
    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, dimensions_vec);
    M.vecmatareisometries = @() true;

end
