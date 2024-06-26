function M = multinomialdoublystochasticgeneralfactory(n, m, p, q)
% Manifold of n-by-m postive matrices such that row sum is p and column sum is q.
%
% function M = multinomialdoublystochasticgeneralfactory(n, m, p, q)
%
%  X > 0.
%  X1 = p, p is a column positive vector of size n.
%  X'1 = q, q is a column positive vector of size m.
% 
% Ensure that p > 0 and q > 0. Also, ensure that sum(p) == sum(q).
%
%
% Please cite the Manopt paper as well as the research papers:
%
%
% @Techreport{mishra21a,
%   Title   = {Manifold optimization for optimal transport},
%   Author  = {Mishra, B. and Satya Dev, N. T. V., Kasai, H. and Jawanpuria, P.},
%   Journal = {Arxiv preprint arXiv:2103.00902},
%   Year    = {2021}
% }
%
% @article{douik2019manifold,
% title={Manifold optimization over the set of doubly stochastic matrices: A second-order geometry},
%  author={Douik, A. and Hassibi, B.},
%  journal={IEEE Transactions on Signal Processing},
%  volume={67},
%  number={22},
%  pages={5761--5774},
%  year={2019}
%}
%
%
% @article{shi21a,
% title={Coupling matrix manifolds assisted optimization for optimal transport problems},
%  author={Shi, D. and Gao, J. and Hong, X. and Choy, ST. B. and Wang, Z.},
%  journal={Machine Learning},
%  pages={1--26},
%  year={2021}
%b}
%
%
% The factory file extends the factory file
% multinomialdoublystochasticfactory 
% to handle general scaling of rows and columns.
%
%
% See also multinomialdoublystochastic multinomialsymmetricfactory multinomialfactory

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Oct 30, 2020.
% Contributors:
% Change log:

    e1 = ones(n, 1);
    e2 = ones(m, 1);

    maxDSiters = min(1000, n*m); % Ideally it should be supplid by user. 

    if size(p, 1) ~= n
        error('p should be a column vector of size n.');
    end

    if size(q, 1) ~= m
        error('q should be a column vector of size m.');
    end

    function [alpha, beta] = mylinearsolve(X, b) % BM okay
        % zeta = sparse(A)\b; % sparse might not be better perf.-wise.
        % where A = [diag(p) X ; X' diag(q)];
        %
        % Even faster is to create a function handle
        % computing A*x (x is a given vector). 
        % Make sure that A is not created, and X is only 
        % passed with mylinearsolve and not A.
        [zeta, ~, ~, iter] = pcg(@mycompute, b, 1e-6, 100);
        function Ax = mycompute(x) % BM okay
            xtop = x(1:n,1); % vector of size n akin to alpha
            xbottom = x(n+1:end,1); % vector of size m akin to beta
            Axtop = xtop.*p + X*xbottom;
            Axbottom = X'*xtop + xbottom.*q;
            Ax = [Axtop; Axbottom];
        end
        alpha = zeta(1:n, 1);
        beta = zeta(n+1:end, 1);
    end

    M.name = @() sprintf('%dx%d matrices with positive entries such that row sum is p and column sum is q', n, n);

    M.dim = @() (n-1)*(m-1); % BM okay

    % Fisher metric
    M.inner = @iproduct; % BM okay
    function ip = iproduct(X, eta, zeta)
        ip = sum((eta(:).*zeta(:))./X(:));
    end

    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta)); % BM okay

    % The manifold is not compact as a result of the choice of the metric,
    % thus any choice here is arbitrary. This is notably used to pick
    % default values of initial and maximal trust-region radius in the
    % trustregions solver.
    M.typicaldist = @() m+n;

    % Pick a random point on the manifold
    M.rand = @random; % BM okay
    function X = random()
        Z = abs(randn(n, m));     % Random point in the ambient space
        X = doubly_stochastic_general(Z, p, q, maxDSiters); % Projection on the Manifold
    end

    % Pick a random vector in the tangent space at X.
    M.randvec = @randomvec; % BM okay
    function eta = randomvec(X) % A random vector in the tangent space
        % A random vector in the ambient space
        Z = randn(n, m);
        % Projection of the vector onto the tangent space
        b = [sum(Z, 2) ; sum(Z, 1)'];
        [alpha, beta] = mylinearsolve(X, b);
        eta = Z - (alpha*e2' + e1*beta').*X;
        % Normalizing the vector
        nrm = M.norm(X, eta);
        eta = eta / nrm;
    end

    % Projection of vector eta in the ambient space to the tangent space.
    M.proj = @projection;  % BM okay
    function etaproj = projection(X, eta) % Projection of the vector eta in the ambeint space onto the tangent space
        b = [sum(eta, 2) ; sum(eta, 1)'];
        [alpha, beta] = mylinearsolve(X, b);
        etaproj = eta - (alpha*e2' + e1*beta').*X;
    end

    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta; % BM okay

    % Conversion of Euclidean to Riemannian gradient
    M.egrad2rgrad = @egrad2rgrad; % BM okay
    function rgrad = egrad2rgrad(X, egrad) % projection of the euclidean gradient
        mu = (X.*egrad); 
        b = [sum(mu, 2) ; sum(mu, 1)'];
        [alpha, beta] = mylinearsolve(X, b);
        rgrad = mu - (alpha*e2' + e1*beta').*X;
    end

    % First-order retraction
    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        Y = X.*exp(t*(eta./X));

        Y = min(Y, 1e50); % For numerical stability
        Y = max(Y, 1e-50); % For numerical stability

        Y = doubly_stochastic_general(Y, p, q, maxDSiters);
    end

    % Conversion of Euclidean to Riemannian Hessian
    M.ehess2rhess = @ehess2rhess; % BM okay
    function rhess = ehess2rhess(X, egrad, ehess, eta)

        % Computing the directional derivative of the Riemannian
        % gradient
        gamma = egrad.*X;
        gammadot = ehess.*X + egrad.*eta;
        
        b = [sum(gamma, 2) ; sum(gamma, 1)'];
        bdot = [sum(gammadot, 2) ; sum(gammadot, 1)'];
        [alpha, beta] = mylinearsolve(X, b);
        [alphadot, betadot] = mylinearsolve(X, bdot- [eta*beta; eta'*alpha]);
        
        S = (alpha*e2' + e1*beta');
        deltadot = gammadot - (alphadot*e2' + e1*betadot').*X- S.*eta; % rgraddot

        % Computing Riemannian gradient
        delta = gamma - S.*X; % rgrad

        % Riemannian Hessian in the ambient space
        nabla = deltadot - 0.5*(delta.*eta)./X;

        % Riemannian Hessian on the tangent space
        rhess = projection(X, nabla);
    end


    % Miscellaneous manifold functions % BM okay
    M.hash = @(X) ['z' hashmd5(X(:))];
    M.lincomb = @matrixlincomb;
    M.zerovec = @(X) zeros(n, m);
    M.transp = @(X1, X2, d) projection(X2, d);
    M.vec = @(X, U) U(:);
    M.mat = @(X, u) reshape(u, n, m);
    M.vecmatareisometries = @() false;
    
end
