function M = sympositivedefinitefactory(n)
% Manifold of n-by-n symmetric positive definite matrices with
% the bi-invariant geometry.
%
% function M = sympositivedefinitefactory(n)
%
% A point X on the manifold is represented as a symmetric positive definite
% matrix X (nxn).
%
% The following material is referenced from Chapter 6 of the book:
% Rajendra Bhatia, "Positive definite matrices",
% Princeton University Press, 2007.

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, August 29, 2013.
% Contributors:
% Change log:
    
    M.name = @() sprintf('Symmetric positive definite geometry of %dx%d matrices', n, n);
    
    M.dim = @() n*(n-1)/2;
    
    % Choice of the metric on the orthnormal space is motivated by the
    % symmetry present in the space. The metric on the positive definite
    % cone is its natural bi-invariant metric.
    M.inner = @(X, eta, zeta) trace( (X\eta) * (X\zeta) );
    
    M.norm = @(X, eta) norm(X\eta, 'fro');
    
    M.dist = @dist;
    function d = dist(X, Y)
        d = norm(logm(X\Y), 'fro');
    end
    
    
    M.typicaldist = @() sqrt(n*(n-1)/2);
    
    symm = @(X) .5*(X+X');
    
    M.egrad2rgrad = @egrad2rgrad;
    function eta = egrad2rgrad(X, eta)
        eta = X*symm(eta)*X;
    end
    
    
    M.ehess2rhess = @ehess2rhess;
    function Hess = ehess2rhess(X, egrad, ehess, eta)
        % Directional derivatives of the Riemannian gradient
        Hess = X*symm(ehess)*X + 2*symm(eta*symm(egrad)*X);
        
        % Correction factor for the non-constant metric
        Hess = Hess - symm(eta*symm(egrad)*X);
    end
    
    
    M.proj = @projection;
    function etaproj = projection(X, eta)
        % Projection onto the tangent space of the total sapce
        etaproj = symm(eta);
    end
    
    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;
    
    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        Y = exponential(X, eta, t);
    end
    
    M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        L = chol(X);
        Y = real(L'*expm(L'\(t*eta)/L)*L);
    end
    
    M.log = @logarithm;
    function U = logarithm(X, Y)
        L = chol(X);
        U = real(L'*logm(L'\Y/L)*L);
    end
    
    M.hash = @(X) ['z' hashmd5(X(:))];
    
    M.rand = @random;
    
    function X = random()
        D = diag(1+rand(n, 1));
        [Q R] = qr(randn(n)); %#ok<NASGU>
        X = Q*D*Q';
    end
    
    M.randvec = @randomvec;
    function eta = randomvec(X)
        % A random vector on the tangent space
        eta = randn(n, n);
        eta = projection(X, eta);
        nrm = M.norm(X, eta);
        eta = eta / nrm;
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(X) zeros(n, n);
    
    M.transp = @(X1, X2, d) projection(X2, d);
    
    % vec and mat are not isometries, because of the unusual inner metric.
    M.vec = @(X, U) U(:);
    M.mat = @(X, u) reshape(u, n, n);
    M.vecmatareisometries = @() false;
    
end

% Linear combination of tangent vectors
function d = lincomb(X, a1, d1, a2, d2) %#ok<INUSL>
    if nargin == 3
        d = a1*d1;
    elseif nargin == 5
        d = a1*d1 + a2*d2;
    else
        error('Bad use of sympositivedefinitefactory.lincomb.');
    end 
end

