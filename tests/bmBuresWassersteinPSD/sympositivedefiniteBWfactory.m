function M = sympositivedefiniteBWfactory(n)
% Manifold of n-by-n symmetric positive definite matrices with
% the Bures-Wassterstein geometry.
%
% function M = sympositivedefiniteBWfactory(n)
%
% A point X on the manifold is represented as a symmetric positive definite
% matrix X (nxn). Tangent vectors are symmetric matrices of the same size
% (but not necessarily definite).
%


% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, January 23, 2020.
% Contributors:
% Change log:
    
    
    symm = @(X) .5*(X+X');
    
    M.name = @() sprintf('Symmetric positive definite geometry of %dx%d matrices with the Wasserstein metric', n, n);
    
    M.dim = @() n*(n+1)/2;
    
    % Helpers to avoid computing full matrices simply to extract their trace
    vec  = @(A) A(:);
    trAB = @(A, B) vec(A')'*vec(B);  % = trace(A*B)
    trAA = @(A) sqrt(trAB(A, A));    % = sqrt(trace(A^2))
    
    % Choice of the metric on the orthonormal space is motivated by the
    % symmetry present in the space. The metric on the positive definite
    % cone is the Bures-Wasserstein metric.
    M.inner = @myinner;
    function ip = myinner(X, eta, zeta)
        ip = 0.5*trAB(symm(lyap(X, -eta)), zeta); % BM: okay
    end
    
    M.norm = @(X, eta) sqrt(myinner(X, eta, eta));
    
    M.dist = @mydist;
    function d = mydist(X, Y)
        Xhalf = sqrtm(X);
        d = sqrt(trace(X) + trace(Y) - 2*trace(symm(sqrtm(Xhalf*Y*Xhalf))));
    end
    
    
    
    M.typicaldist = @() sqrt(n*(n+1)/2); % BM: okay    
    
    M.egrad2rgrad = @egrad2rgrad;
    function eta = egrad2rgrad(X, eta)
        eta = 4*symm(eta*X);
    end
    
    
    M.ehess2rhess = @ehess2rhess;
    function Hess = ehess2rhess(X, egrad, ehess, eta)
        % Directional derivatives of the Riemannian gradient
        Hess = 4*symm(ehess*X) + 4*symm(egrad*eta);
        
        % Correction factor for the non-constant BW metric
        rgrad = egrad2rgrad(X, egrad);
        rgrad1 = lyap(X, -rgrad);
        eta1 = lyap(X, -eta);
        Hess = Hess ...
            - symm(rgrad1 * eta) ...
            - symm(rgrad * eta1) ...
            + 2*symm(X*symm(rgrad1 * eta1));
    end
    
    
    M.proj = @(X, eta) symm(eta);
    
    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;
    
    M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        teta = t*eta;
        teta1 = symm(lyap(X, -teta));
        Y = X + teta + teta1*X*teta1;
    end
    
    M.retr = @exponential;
    
    function ABhalf = myhalf(A, B)
        Ahalf = sqrtm(A);
        ABhalf = (Ahalf*symm(sqrtm(Ahalf*B*Ahalf)))/Ahalf;
    end
    
    M.log = @logarithm;
    function H = logarithm(X, Y)
        H = symm(myhalf(X, Y) + myhalf(Y, X) - 2*X);
    end
    
    
    
    M.hash = @(X) ['z' hashmd5(X(:))];
    
    % Generate a random symmetric positive definite matrix following a
    % certain distribution. The particular choice of a distribution is of
    % course arbitrary, and specific applications might require different
    % ones.
    M.rand = @random;
    function X = random()
        D = diag(1+rand(n, 1));
        [Q, R] = qr(randn(n)); %#ok
        X = Q*D*Q';
    end
    
    % Generate a uniformly random unit-norm tangent vector at X.
    M.randvec = @randomvec;
    function eta = randomvec(X)
        eta = symm(randn(n));
        nrm = M.norm(X, eta);
        eta = eta / nrm;
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(X) zeros(n);
    
    % Poor man's vector transport: exploit the fact that all tangent spaces
    % are the set of symmetric matrices, so that the identity is a sort of
    % vector transport. It may perform poorly if the origin and target (X1
    % and X2) are far apart though. This should not be the case for typical
    % optimization algorithms, which perform small steps.
    M.transp = @(X1, X2, eta) eta;
    
    % vec and mat are not isometries, because of the unusual inner metric.
    M.vec = @(X, U) U(:);
    M.mat = @(X, u) reshape(u, n, n);
    M.vecmatareisometries = @() false;
    
end
