function M = multinomialsymmetricfactory(n)
% Manifold of n-by-n symmetric stochastic matrices with positive entries.
%
% function M = multinomialsymmetricfactory(n) 
%
% The returned structure M is a Manopt manifold structure to optimize over
% the set of n-by-n symmetric matrices with (strictly) positive entries 
% and such that the entries of each column and each row sum to one.
%
% The metric imposed on the manifold is the Fisher metric such that 
% the set of n-by-n symmetric-stochastic matrices is a Riemannian 
% submanifold of the space of symmetric n-by-n matrices. Also it should be 
% noted that the  retraction operation that we define is first order and 
% as such the  checkhessian tool cannot verify he slope correctly unless 
% at the optimum.
%
% If the optimal solution to the problem has vanishing entries, it is not
% achievable by the manifold. Therefore, the checkhessian tool is expected
% to fail even at the optimum.
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
    symm = @(X) .5*(X+X'); % Symmetric part of a matrix

    % Manifold geometry functions

    M.name = @() sprintf('%dx%d symmetric doubly-stochastic matrices with positive entries', n, n);

    M.dim = @() n*(n-1)/2;

    M.inner = @iproduct; % We impose the Fisher metric.
    function ip = iproduct(X, eta, zeta)
        ip = sum((eta(:).*zeta(:))./X(:));
    end

    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));

    M.dist = @(X, Y) error('multinomialsymmetricfactory.dist not implemented yet.');

    M.typicaldist = @() n*pi/2 ;% This is an approximation.
    
    % Random points, vectors and projections

    M.rand = @random;
    function X = random() % A random point in the manifold
        Z = symm(abs(randn(n, n))) ;  % A random point in the ambient space
        X = symm(doubly_stochastic(Z)) ; % Projection on the Manifold
    end

    M.randvec = @randomvec;
    function eta = randomvec(X) % A random vector in the tangent space
        % A random vector in the ambient space
        Z = symm(randn(n, n)) ; 
        % Projection onto the tangent space.
        alpha = sum((eye(n) + X)\(Z),2) ;
        eta = Z - (alpha*e' + e*alpha').*X ;
        % Normalizing the vector
        nrm = M.norm(X, eta);
        eta = eta / nrm;
    end

    M.proj = @projection; 
    function etaproj = projection(X, eta) % Projection of the vector eta in the ambient space onto the tangent space
        alpha = sum((eye(n) + X)\(eta),2) ;
        etaproj = eta - (alpha*e' + e*alpha').*X ;
    end

    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;

    % Gradient, retraction, and exponential map computation

    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad) % Projection of the Euclidean gradient
        mu = sum((X.*egrad),2) ;
        alpha = (eye(n) + X)\mu ;
        rgrad = (egrad - alpha*e' - e*alpha').*X; 
    end

    M.retr = @retraction;
    function Y = retraction(X, eta, t) % Retraction of the tangent vector to the manifold
        if nargin < 3
            t = 1.0;
        end
        Y = X.*exp(t*(eta./X));
        Y = symm(doubly_stochastic(Y)) ;
        Y = max(Y, eps);
    end

    M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        Y = retraction(X, eta, t);
        warning('manopt:multinomialsymmetricfactory:exp', ...
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
        alpha = sum((eye(n) + X)\(gamma),2) ;
        m = (eye(n)+X)\eta ;
        alphadot = sum((eye(n) + X)\(gammadot- m*gamma),2) ;
        S = (alpha*e' + e*alpha') ;
        deltadot = gammadot - (alphadot*e' + e*alphadot').*X- S.*eta ;

        % Projecting gamma
        delta = gamma - S.*X; 

        % Computing and projecting nabla
        nabla = deltadot - 0.5*(delta.*eta)./X ;
        w = sum((eye(n) + X)\(nabla),2) ; 
        rhess = nabla - (w*e' + e*w').*X; 
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
