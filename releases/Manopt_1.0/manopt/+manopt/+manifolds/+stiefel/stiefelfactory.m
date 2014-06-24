function M = stiefelfactory(n, p, k)
% Returns a manifold structure to optimize over orthonormal matrices.
%
% function M = stiefelfactory(n, p)
% function M = stiefelfactory(n, p, k)
%
% The Stiefel manifold is the set of orthonormal nxp matrices. If k
% is larger than 1, this is the Cartesian product of the Stiefel manifold
% taken k times. The metric is such that the manifold is a Riemannian
% submanifold of R^nxp equipped with the usual trace inner product, that
% is, it is the usual metric.
%
% Points are represented as matrices X of size n x p x k (or n x p if k=1,
% which is the default) such that each n x p matrix is orthonormal,
% i.e., X'*X = eye(p) if k = 1, or X(:, :, i)' * X(:, :, i) = eye(p) for
% i = 1 : k if k > 1. Tangent vectors are represented as matrices the same
% size as points.
%
% By default, k = 1.
%
% See also: grassmannfactory rotationsfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 

    
    if ~exist('k', 'var') || isempty(k)
        k = 1;
    end
    
    if k == 1
        M.name = @() sprintf('Stiefel manifold St(%d, %d)', n, p);
    elseif k > 1
        M.name = @() sprintf('Product Stiefel manifold St(%d, %d)^%d', n, p, k);
    else
        error('k must be an integer no less than 1.');
    end
    
    M.dim = @() k*(n*p - .5*p*(p+1));
    
    M.inner = @(x, d1, d2) d1(:).'*d2(:);
    
    M.norm = @(x, d) norm(d(:));
    
    M.dist = @(x, y) error('stiefel.dist not implemented yet.');
    
    M.typicaldist = @() sqrt(p*k);
    
    M.proj = @projection;
    function Up = projection(X, U)
        
        XtU = manopt.tools.multiprod(manopt.tools.multitransp(X), U);
        symXtU = .5*(XtU + manopt.tools.multitransp(XtU));
        Up = U - manopt.tools.multiprod(X, symXtU);
        
% The code above is equivalent to, but much faster than, the code below.
%         
%     Up = zeros(size(U));
%     function A = sym(A), A = .5*(A+A'); end
%     for i = 1 : k
%         Xi = X(:, :, i);
%         Ui = U(:, :, i);
%         Up(:, :, i) = Ui - Xi*sym(Xi'*Ui);
%     end

    end
    
    M.retr = @retraction;
    function Y = retraction(X, U, t)
        if nargin < 3
            t = 1.0;
        end
        Y = X + t*U;
        for i = 1 : k
            [Q, R] = qr(Y(:, :, i), 0);
            % The funny business with R assures we are not flipping signs
            % of some columns, which should never happen in modern Matlab
            % versions but may be an issue with older versions.
            Y(:, :, i) = Q * diag(sign(sign(diag(R))+.5));
        end
    end
    
    M.exp = @exponential;
    function Y = exponential(X, U, t)
        if nargin == 2
            t = 1;
        end
        Y = retraction(X, U, t);
        warning('manopt:stiefel:exp', 'Exponential for Stiefel manifold not implemented yet. Used retraction instead.');
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
        error('Bad use of stiefel.lincomb.');
    end

end
