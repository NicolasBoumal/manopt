function M = symposspeciallinear(n)
% Manifold of n-by-n symmetric special linear matrices with
% the bi-invariant geometry.
%
% function M = symspeciallinear(n)
%
% A point X on the manifold is represented as a symmetric special linear
% matrix X (nxn) with det(X)=1. Tangent vectors are symmetric matrices of the same size
% which are trace orthogonal to inv(X), i.e. tr(inv(X)*eta)=0.
%
% The Riemannian metric is the bi-invariant metric, described notably in
% Chapter 6 of the 2007 book "Positive definite matrices"
% by Rajendra Bhatia, Princeton University Press.
%
%
% This file is part of Manopt: www.manopt.org.
% Original author: Alexander Müller, January 26, 2022.
% Contributors:
% Change log:
%


symm = @(X) .5*(X+X.');

M.name = @() sprintf('Symmetric special linear geometry of %dx%d matrices with unit determinant', n, n);

M.dim = @() n*(n+1)/2-1;

% Helpers to avoid computing full matrices simply to extract their trace
vec  = @(A) A(:);
trAB = @(A, B) vec(A.').'*vec(B);  % = trace(A*B)
trAA = @(A) sqrt(trAB(A, A));    % = sqrt(trace(A^2))


% Helper to normalize Matrix to unit determinant
M.normalizetounitdet = @(A) A/power(det(A),1/n);

% Choice of the metric on the orthonormal space is motivated by the
% symmetry present in the space. The metric on the positive definite
% cone is its natural bi-invariant metric.
% The result is equal to: trace( (X\eta) * (X\zeta) )
% Maybe this is not the best inner product for the subspace of unit
% determinants
M.inner = @(X, eta, zeta) trAB(X\eta, X\zeta);

% Notice that X\eta is *not* symmetric in general.
% The result is equal to: sqrt(trace((X\eta)^2))
% There should be no need to take the real part, but rounding errors
% may cause a small imaginary part to appear, so we discard it.
M.norm = @(X, eta) real(trAA(X\eta));

% Same here: X\Y is not symmetric in general.
% Same remark about taking the real part.
% Maybe this not the correct distance function for the subspace of unit
% determinants
M.dist = @(X, Y) real(trAA(real(logm(X\Y))));


M.typicaldist = @() sqrt(n*(n+1)/2-1);


M.egrad2rgrad = @egrad2rgrad;
    function eta = egrad2rgrad(X, eta)
        eta = X*symm(eta)*X-1/n*trAB(X,symm(eta))*X;        
    end


%This is only the correct Riemannian Hessian for n=2 since it is derived
%from the projection-based retraction normalizetounitdet which is only a
%second order retraction for n=2
M.ehess2rhess = @ehess2rhess;
    function Hess = ehess2rhess(X, egrad, ehess, eta) 
        Hess= X*symm(ehess)*X-1/n*trAB(X,symm(ehess))*X;
        Hess = Hess + 1/n*(  trAB((X),symm(egrad))*M.proj(X,(eta)));
    end


M.proj = @(X, eta) symm(eta)-1/n*trace(X\symm(eta))*X;
M.tangent = M.proj;
M.tangent2ambient = @(X, eta) eta;


% This retraction is only second order for n=2
% For larger matrices it is only a first order retraction
M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            teta = eta;
        else
            teta = t*eta;
        end
        Y = M.normalizetounitdet(symm(X + teta));
    end

M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        % The symm() and real() calls are mathematically not necessary but
        % are numerically necessary.
        Y = symm(X*real(expm(X\(t*eta))));
    end

M.log = @logarithm;
    function H = logarithm(X, Y)
        % Same remark regarding the calls to symm() and real().
        H = symm(X*real(logm(X\Y)));
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
        X = Q*D*Q.';
        X = M.normalizetounitdet(X);
    end

% Generate a uniformly random unit-norm tangent vector at X.
M.randvec = @randomvec;
    function eta = randomvec(X)
        eta = symm(randn(n));
        nrm = M.norm(X, eta);
        eta = eta / nrm;
        eta= M.proj(X,eta);
    end

M.lincomb = @matrixlincomb;

M.zerovec = @(X) zeros(n);

% Poor man's vector transport: exploit the fact that all tangent spaces
% are the set of symmetric matrices, so that the identity is a sort of
% vector transport. It may perform poorly if the origin and target (X1
% and X2) are far apart though. This should not be the case for typical
% optimization algorithms, which perform small steps.
M.transp = @(X1, X2, eta) eta;

% For reference, a proper vector transport is given here, following
% work by Sra and Hosseini: "Conic geometric optimisation on the
% manifold of positive definite matrices", in SIAM J. Optim.
% in 2015; also available here: http://arxiv.org/abs/1312.1039
% This will not be used by default. To force the use of this transport,
% execute "M.transp = M.paralleltransp;" on your M returned by the
% present factory.
M.paralleltransp = @parallel_transport;
    function zeta = parallel_transport(X, Y, eta)
        E = sqrtm(Y/X);
        zeta = E*eta*E.';
    end

% vec and mat are not isometries, because of the unusual inner metric.
M.vec = @(X, U) U(:);
M.mat = @(X, u) reshape(u, n, n);
M.vecmatareisometries = @() false;

end
