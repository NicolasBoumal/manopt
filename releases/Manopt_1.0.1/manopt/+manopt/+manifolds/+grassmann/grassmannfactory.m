function M = grassmannfactory(n, p, k)
% Returns a manifold struct to optimize over the space of vector subspaces.
%
% function M = grassmannfactory(n, p)
% function M = grassmannfactory(n, p, k)
%
% Grassmann manifold: each point on this manifold is a collection of k
% vector subspaces of dimension p embedded in R^n.
%
% The metric is obtained by making the Grassmannian a Riemannian quotient
% manifold of the Stiefel manifold, i.e., the manifold of orthonormal
% matrices, itself endowed with a metric by making it a Riemannian
% submanifold of the Euclidean space, endowed with the usual trace inner
% product. In short: it is the usual metric used in most cases.
% 
% This structure deals with matrices X of size n x p x k (or n x p if
% k = 1, which is the default) such that each n x p matrix is orthonormal,
% i.e., X'*X = eye(p) if k = 1, or X(:, :, i)' * X(:, :, i) = eye(p) for
% i = 1 : k if k > 1. Each n x p matrix is a numerical representation of
% the vector subspace its columns span.
%
% This function has been little tested.
%
% By default, k = 1.
%
% See also: stiefelfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    if ~exist('k', 'var') || isempty(k)
        k = 1;
    end
    
    if k == 1
        M.name = @() sprintf('Grassmann manifold Gr(%d, %d)', n, p);
    elseif k > 1
        M.name = @() sprintf('Multi Grassmann manifold Gr(%d, %d)^%d', ...
                             n, p, k);
    else
        error('k must be an integer no less than 1.');
    end
    
    M.dim = @() k*p*(n-p);
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:);
    
    M.norm = @(x, d) norm(d(:));
    
    M.dist = @(x, y) error('grassmann.dist not implemented yet.');
    
    M.typicaldist = @() sqrt(p*k);
    
    % TODO: this is projection on the horizontal space; make this clear in
    % the documentation that this is what is intended.
    M.proj = @projection;
    function Up = projection(X, U)
        
        XtU = manopt.tools.multiprod(manopt.tools.multitransp(X), U);
        Up = U - manopt.tools.multiprod(X, XtU);

    end
    
	M.egrad2rgrad = M.proj;
    
    M.retr = @retraction;
    function Y = retraction(X, U, t)
        if nargin < 3
            t = 1.0;
        end
        Y = X + t*U;
        for i = 1 : k
            % We do not need to worry about flipping signs of columns here,
            % since only the column space is important, not the actual
            % columns. Compare this with the Stiefel manifold.
            [Q, unused] = qr(Y(:, :, i), 0); %#ok
            Y(:, :, i) = Q;
        end
    end
    
    M.exp = @exponential;
    function Y = exponential(X, U, t)
        if nargin == 2
            t = 1;
        end
        Y = retraction(X, U, t);
        warning('manopt:grassmann:exp', 'Exponential for Grassmann manifold not implemented yet. Used retraction instead.');
    end

    M.hash = @(X) ['z' manopt.privatetools.hashmd5(X(:))];
    
    M.rand = @random;
    function X = random()
        X = zeros(n, p, k);
        for i = 1 : k
            % TODO: check that this is correct
            [Q, unused] = qr(randn(n, p), 0); %#ok
            X(:, :, i) = Q; %Q(:, 1:p);
%             X(:, :, i) = X(:, :, i)*(X(:, :, i)'*X(:, :, i))^-.5;
        end
    end
    
    M.randvec = @randomvec;
    function U = randomvec(X)
        U = projection(X, randn(n, p, k));
        for i = 1 : k
            U(:, :, i) = U(:, :, i) / norm(U(:, :, i), 'fro');
        end
        U = U / sqrt(k);
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(x) zeros(n, p, k);
    
    M.transp = @(x1, x2, d) projection(x2, d);

end

% Linear combination of tangent vectors
function d = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>

    if nargin == 3
        d = a1*d1;
    elseif nargin == 5
        d = a1*d1 + a2*d2;
    else
        error('Bad use of grassmann.lincomb.');
    end

end
