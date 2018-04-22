function M = multinomialdoublystochasticfactory(n)
% Manifold of n-by-n doubly-stochastic matrices with positive entries.
%
% function M = multinomialdoublystochasticfactory(n)
%
% The returned structure M is a Manopt manifold structure to optimize over
% the set of n-by-n matrices with (strictly) positive entries and such that
% the entries of each column and each row sum to one.
%
% The metric imposed on the manifold is the Fisher metric such that
% the set of n-by-n doubly-stochastic matrices is a Riemannian submanifold
% of the space of n-by-n matrices. Also it should be noted that the
% retraction operation that we define is first order and as such the
% checkhessian tool cannot verify he slope correctly unless at the optimum.
%
% If the optimal solution to the problem has vanishing entries, it is not
% achievable by the manifold. Therefore, the checkhessian tool is expected
% to fail even at the optimum.
%
% The file is based on developments in the research paper
% A. Douik and B. Hassibi, "Manifold Optimization Over the Set
% of Doubly Stochastic Matrices: A Second-Order Geometry"
% ArXiv:1802.02628, 2018.
%
% Link to the paper: https://arxiv.org/abs/1802.02628.
%
% Please cite the Manopt paper as well as the research paper:
%     @Techreport{Douik2018Manifold,
%       Title   = {Manifold Optimization Over the Set of Doubly Stochastic
%		   Matrices: {A} Second-Order Geometry},
%       Author  = {Douik, A. and Hassibi, B.},
%       Journal = {Arxiv preprint ArXiv:1802.02628},
%       Year    = {2018}
%     }

% This file is part of Manopt: www.manopt.org.
% Original author: Ahmed Douik, March 06, 2018.
% Contributors:
% Change log:

    % Variable and functions
    e = ones(n, 1); % Column vector of ones of length n.

    % Manifold geometry functions

    M.name = @() sprintf('%dx%d doubly-stochastic matrices with positive entries', n, n);

    M.dim = @() (n-1)^2;

    M.inner = @iproduct; % We impose the Fisher metric.
    function ip = iproduct(X, eta, zeta)
        ip = sum((eta(:).*zeta(:))./X(:));
    end

    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));

    M.dist = @(X, Y) error('dsmultinomialfactory.dist not implemented yet.');

    M.typicaldist = @() n*pi/2 ;% This is an approximation.

    % Random points, vectors and projections

    M.rand = @random;
    function X = random() % A random point in the manifold
        Z = abs(randn(n, n)) ;  % A random point in the ambient space
        X = doubly_stochastic(Z) ; % Projection on the Manifold
    end

    M.randvec = @randomvec;
    function eta = randomvec(X) % A random vector in the tangent space
        % A random vector in the ambient space
        Z = randn(n, n) ;
        % Projection of the vector onto the tangent space
        zeta = pinv([eye(n) X ; X' eye(n)])*[sum(Z,2) ; sum(Z,1)'];
        alpha = zeta(1:n) ;
        beta = zeta(n+1:2*n) ;
        eta = Z - (alpha*e' + e*beta').*X ;
        % Normalizing the vector
        nrm = M.norm(X, eta);
        eta = eta / nrm;
    end

    M.proj = @projection;
    function etaproj = projection(X, eta) % Projection of the vector eta in the ambient space onto the tangent space
        alpha = sum(pinv(eye(n)-X*X')*(eta-X*eta'),2);
        beta = sum(eta,1)' - X'*alpha ;
        etaproj = eta - (alpha*e' + e*beta').*X ;
    end

    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;

    % Gradient, retraction, and exponential map computation

    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad) % Projection of the Euclidean gradient
        mu = (X.*egrad) ;
        alpha = sum(pinv(eye(n)-X*X')*(mu-X*mu'),2);
        beta = sum(mu,1)' - X'*alpha ;
        rgrad = mu - (alpha*e' + e*beta').*X ;
    end

    M.retr = @retraction;
    function Y = retraction(X, eta, t) % Retraction of the tangent vector to the manifold
        if nargin < 3
            t = 1.0;
        end
        Y = X.*exp(t*(eta./X));
        Y = doubly_stochastic(Y) ;
        Y = max(Y, eps);
    end

    M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        Y = retraction(X, eta, t);
        warning('manopt:dsmultinomialfactory:exp', ...
            ['Exponential for the Multinomial manifold' ...
            'manifold not implemented yet. Used retraction instead.']);
    end

    % Hessian computation

    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, eta)

        % Computing the direcitonal derivative of the riemannian
        % gradient
        gamma = egrad.*X ;
        gammadot = ehess.*X + egrad.*eta ;
        epsilon = pinv([eye(n) X ; X' eye(n)]) ;
        epsilondot = -epsilon*[zeros(n,n) eta ; eta' zeros(n,n)]*epsilon ;
        zeta = epsilon*[sum(gamma,2) ; sum(gamma,1)'];
        alpha = zeta(1:n) ;
        beta = zeta(n+1:2*n) ;
        zetadot = epsilondot*[sum(gamma,2) ; sum(gamma,1)'] + epsilon*[sum(gammadot,2) ; sum(gammadot,1)'];
        alphadot = zetadot(1:n) ;
        betadot = zetadot(n+1:2*n) ;

        S = (alpha*e' + e*beta') ;

        deltadot = gammadot - (alphadot*e' + e*betadot').*X- S.*eta ;

        % Projecting gamma
        delta = gamma - S.*X;

        % Computing and projecting nabla
        nabla = deltadot - 0.5*(delta.*eta)./X ;

        zeta = pinv([eye(n) X ; X' eye(n)])*[sum(nabla,2) ; sum(nabla,1)'];
        alpha = zeta(1:n) ;
        beta = zeta(n+1:2*n) ;
        rhess = nabla - (alpha*e' + e*beta').*X;
    end


    % Miscellaneous manifold functions
    M.hash = @(X) ['z' hashmd5(X(:))];
    M.lincomb = @matrixlincomb;
    M.zerovec = @(X) zeros(n, n);
    M.transp = @(X1, X2, d) projection(X2, d);
    M.vec = @(X, U) U(:);
    M.mat = @(X, u) reshape(u, n, n);
    M.vecmatareisometries = @() false;
    
end
