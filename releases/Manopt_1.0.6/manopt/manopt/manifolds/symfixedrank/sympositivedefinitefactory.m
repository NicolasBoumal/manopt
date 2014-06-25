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
% Contributors: Nicolas Boumal
% Change log:
%
%   March 5, 2014 (NB)
%       There were a number of mistakes in the code owing to the tacit
%       assumption that if X and eta are symmetric, then X\eta is
%       symmetric too, which is not the case. See discussion on the Manopt
%       forum started on Jan. 19, 2014. Functions norm, dist, exp and log
%       were modified accordingly. Furthermore, they only require matrix
%       inversion (as well as matrix log or matrix exp), not matrix square
%       roots or their inverse.
% 
    
    symm = @(X) .5*(X+X');
    
    M.name = @() sprintf('Symmetric positive definite geometry of %dx%d matrices', n, n);
    
    M.dim = @() n*(n-1)/2;
    
    % Choice of the metric on the orthnormal space is motivated by the
    % symmetry present in the space. The metric on the positive definite
    % cone is its natural bi-invariant metric.
    M.inner = @(X, eta, zeta) trace( (X\eta) * (X\zeta) );
    
    % Notice that X\eta is *not* symmetric in general.
    M.norm = @(X, eta) sqrt(trace((X\eta)^2));
    
    % Same here: X\Y is not symmetric in general. There should be no need
    % to take the real part, but rounding errors may cause a small
    % imaginary part to appear, so we discard it.
    M.dist = @(X, Y) sqrt(real(trace((logm(X\Y))^2)));
    
    
    M.typicaldist = @() sqrt(n*(n-1)/2);
    
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
    function etaproj = projection(X, eta) %#ok<INUSL>
        % Projection onto the tangent space of the total space
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
        Y = X*real(expm(X\(t*eta)));
    end
    
    M.log = @logarithm;
    function H = logarithm(X, Y)
        H = X*real(logm(X\Y));
    end
    
    M.hash = @(X) ['z' hashmd5(X(:))];
    
    % Generate a random symmetric positive definite matrix following a
    % certain distribution. The particular choice of a distribution is of
    % course arbitrary, and specific applications might require different
    % ones.
    M.rand = @random;
    function X = random()
        D = diag(1+rand(n, 1));
        [Q R] = qr(randn(n)); %#ok<NASGU>
        X = Q*D*Q';
    end
    
    % Generate a uniformly random unit-norm tangent vector at X.
    M.randvec = @randomvec;
    function eta = randomvec(X)
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

