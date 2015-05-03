function M = complexstiefelfactory(n, p, k)
% Returns a manifold structure to optimize over complex orthonormal matrices.
%
% function M = complexstiefelfactory(n, p)
% function M = complexstiefelfactory(n, p, k)
%
% The complex Stiefel manifold is the set of complex orthonormal nxp matrices. If k
% is larger than 1, this is the Cartesian product of the complex Stiefel manifold
% taken k times. The metric is such that the manifold is a Riemannian
% submanifold of C^nxp equipped with the usual trace inner product, that
% is, it is the usual metric.
%
% Points are represented as matrices X of size n x p x k (or n x p if k=1,
% which is the default) such that each complex n x p matrix is orthonormal,
% i.e., X'*X = eye(p) if k = 1, or X(:, :, i)' * X(:, :, i) = eye(p) for
% i = 1 : k if k > 1. Tangent vectors are represented as matrices the same
% size as points.
%
% By default, k = 1.
%
% See also: stiefelfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Hiroyuki Sato, April 27, 2015.
% Contributors: 
% Change log: 
    
    if ~exist('k', 'var') || isempty(k)
        k = 1;
    end
    
    if k == 1
        M.name = @() sprintf('Complex Stiefel manifold St(%d, %d)', n, p);
    elseif k > 1
        M.name = @() sprintf('Product complex Stiefel manifold St(%d, %d)^%d', n, p, k);
    else
        error('k must be an integer no less than 1.');
    end
    
    M.dim = @() k*(2*n*p - p^2); %!
    
    M.inner = @(x, d1, d2) real(d1(:)'*d2(:)); %!
    
    M.norm = @(x, d) norm(d(:));
    
    M.dist = @(x, y) error('stiefel.dist not implemented yet.');
    
    M.typicaldist = @() sqrt(p*k);
    
    M.proj = @projection;
    function Up = projection(X, U)
        
        XHU = multiprod(multihconj(X), U); %! XtU -> XHU, multitransp -> multihconj
        herXHU = multiher(XHU); %! symXtU -> herXHU, multisym -> multiher
        Up = U - multiprod(X, herXHU); %! symXtU -> herXHU
        
        % The code above is equivalent to, but much faster than, the code below.
        %
        %     Up = zeros(size(U));
        %     function A = her(A), A = .5*(A+A'); end
        %     for i = 1 : k
        %         Xi = X(:, :, i);
        %         Ui = U(:, :, i);
        %         Up(:, :, i) = Ui - Xi*her(Xi'*Ui);
        %     end
        
    end
    
    M.tangent = M.proj;
    
    % For Riemannian submanifolds, converting a Euclidean gradient into a
    % Riemannian gradient amounts to an orthogonal projection.
	M.egrad2rgrad = M.proj;
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, H)
        XHG = multiprod(multihconj(X), egrad); %! XtG -> XHG, multitransp -> multihconj
        herXHG = multiher(XHG); %! symXtG -> herXHG, multisym(XtG) -> multiher(XHG)
        HherXHG = multiprod(H, herXHG); %! HsymXtG -> HherXHG, symXtG -> herXHG
        rhess = projection(X, ehess - HherXHG); %! HsymXtG -> HherXHG
    end
    
    M.retr = @retraction;
    function Y = retraction(X, U, t)
        if nargin < 3
            t = 1.0;
        end
        Y = X + t*U;
        for i = 1 : k
            [Q, R] = qr(Y(:, :, i), 0);
            % The instruction with R assures we are not flipping signs
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
        tU = t*U;
        Y = zeros(size(X));
        for i = 1 : k
            % From a formula by Ross Lippert, Example 5.4.2 in AMS08.
            Xi = X(:, :, i);
            Ui = tU(:, :, i);
            Y(:, :, i) = [Xi Ui] * ...
                         expm([Xi'*Ui , -Ui'*Ui ; eye(p) , Xi'*Ui]) * ...
                         [ expm(-Xi'*Ui) ; zeros(p) ];
        end
        
    end

    M.hash = @(X) ['z' hashmd5([real(X(:)) ; imag(X(:))])]; %! X(:) -> [real(X(:)) ; imag(X(:))]
    
    M.rand = @random;
    function X = random()
        X = zeros(n, p, k);
        for i = 1 : k
            [Q, unused] = qr(randn(n, p) + 1i*randn(n,p), 0); %! Complex version
            X(:, :, i) = Q;
        end
    end
    
    M.randvec = @randomvec;
    function U = randomvec(X)
        U = projection(X, randn(n, p, k) + 1i*randn(n, p, k)); %! Complex version
        U = U / norm(U(:));
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(x) zeros(n, p, k);
    
    M.transp = @(x1, x2, d) projection(x2, d);
    
    M.vec = @(x, u_mat) u_mat(:);
    M.mat = @(x, u_vec) reshape(u_vec, [n, p, k]);
    M.vecmatareisometries = @() true;

end

% Linear combination of tangent vectors
function d = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>

    if nargin == 3
        d = a1*d1;
    elseif nargin == 5
        d = a1*d1 + a2*d2;
    else
        error('Bad use of complexstiefel.lincomb.');
    end

end
