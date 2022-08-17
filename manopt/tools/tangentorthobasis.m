function orthobasis = tangentorthobasis(M, x, n, basisvecs)
% Returns an orthonormal basis of tangent vectors in the Manopt framework.
%
% function orthobasis = tangentorthobasis(M, x)
% function orthobasis = tangentorthobasis(M, x, n)
% function orthobasis = tangentorthobasis(M, x, n, basisvecs)
%
% M is a Manopt manifold structure obtained from a factory.
% x is a point on the manifold M.
% n (optional) is the dimension of the random subspace to span; by default,
%   n = M.dim() so that the returned basis spans the whole tangent space.
% basisvecs (optional) is a cell of tangent vectors that are assumed to be
%   linearly independent. 
%
% orthobasis is a cell of n tangent vectors at x.
% With high probability, they form an orthonormal basis of the tangent
% space at x. If necessary, this can be checked by calling
%   G = grammatrix(M, x, orthobasis)
% and verifying that norm(G - eye(size(G))) is close to zero.
%
% When basisvecs is not specified then n vectors will be taken at random 
% in the tangent space and orthonormalized using Gram-Schmidt.
%
% When basisvecs is specified then the first min(n, length(basisvecs))
% vectors in basisvecs is passed to orthogonalize, with n - length(basisvecs)
% additional randomly sampled tangent vectors if length(basisvecs) < n.
%
% Therefore if basisvecs is provided, the span of the first 
% min(n, length(basisvecs)) vectors in basisvecs will always be a 
% subspace of the space spaned by the returned orthobasis.
%
% Note: if extra accuracy is required, it may help to re-orthogonalize the
% basis returned by this function once, as follows:
%  B = tangentorthobasis(M, x, n);
%  B = orthogonalize(M, x, B);
%
% See also: grammatrix orthogonalize lincomb tangent2vec plotprofile

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, April 28, 2016.
% Contributors: 
% Change log: 
%
%   VL July 17, 2022:
%       Added the option to input basisvecs to specify a linearly
%       independent set of tangent vectors to pass to orthogonalize.


    dim = M.dim();
    if ~exist('n', 'var') || isempty(n)
        n = dim;
    end
    assert(n >= 0 && n <= dim && n == round(n), ...
           'n must be an integer between 0 and M.dim().');
    
    basis = cell(n, 1);
    
    % With high probability, vectors taken at random in the tangent space
    % are linearly independent of basisvecs
    for k = 1 : n
        if exist('basisvecs', 'var') && k <= length(basisvecs)
            basis(k, 1) = basisvecs(k, 1);
        else
            basis{k} = M.randvec(x);
        end
    end

    % The Gram-Schmidt process transforms any n linearly independent
    % vectors into n orthonormal vectors spanning the same subspace.
    orthobasis = orthogonalize(M, x, basis);
    
end
