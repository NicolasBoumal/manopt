function M = sympositivedefinitesimplexfactory(n, k)
% with the bi-invariant geometry such that the sum is the identity matrix.
%
% function M = sympositivedefinitesimplexfactory(n, k)
%
% Given X1, X2, ... Xk symmetric positive definite matrices, the constraint
% tackled is
% X1 + X2 + ... = I.
%
% The Riemannian structure enforced on the manifold 
% M:={(X1, X2,...) : X1 + X2 + ... = I } is a submanifold structure of the 
% total space defined as the k Cartesian product of symmetric positive 
% definite Riemannian manifold (of n-by-n matrices) endowed with the bi-invariant metric.
%
% A point X on the manifold is represented as multidimensional array
% of size n-by-n-by-k. Each n-by-n matrix is symmetric positive definite.
%
% The embedding space is the k Cartesian product of matrices of size
% n-by-n (symmetry not required). The Euclidean gradient and Hessian expressions 
% needed for egrad2rgrad and ehess2rhess are in the embedding space
% endowed with the Euclidean metric.
%
% E = (R^(nxn))^k is the embedding space: we have the obvious representation of points 
% there as 3D arrays of size nxnxk. It is equipped with the standard Euclidean metric.
%
% P = {X in R^(nxn) : X = X' and X positive definite} is a submanifold of R^(nxn). 
% We turn it into a Riemannian manifold (but not a Riemannian submanifold) by equipping
% it with the bi-invariant metric.
%
% M = {X in P^k : X_1 + ... + X_k = I} is the manifold we care about here: it is 
% a Riemannian submanifold of P^k, hence it is also a submanifold (but not a Riemannian
% submanifold) of E -- our embedding space.
%
%
% Please cite the Manopt paper as well as the research paper:
%
%     @techreport{mishra2019riemannian,
%       title={Riemannian optimization on the simplex of positive definite matrices},
%       author={Mishra, B. and Kasai, H. and Jawanpuria, P.},
%       institution={arXiv preprint arXiv:1906.10436},
%       year={2019}
%     }
%
% See also sympositivedefinitesimplexcomplexfactory multinomialfactory sympositivedefinitefactory

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, September 18, 2019.
% Contributors: NB
% Change log:
%   Dec. 16, 2019 (BM): Comments updated
%   Nov.  1, 2021 (BM): Removed typos in Hessian expression
%   June 26, 2024 (NB): Removed M.exp() as it was not implemented.
    
    symm = @(X) .5*(X+X');
    
    M.name = @() sprintf('%d symmetric positive definite matrices of size %dx%d such that their sum is the identiy matrix.', k, n, n);
    
    M.dim = @() (k-1)*n*(n+1)/2;
    
    % Helpers to avoid computing full matrices simply to extract their trace
    vec     = @(A) A(:);
    trinner = @(A, B) vec(A')'*vec(B);  % = trace(A*B)
    trnorm  = @(A) sqrt(trinner(A, A)); % = sqrt(trace(A^2))
        
   
    % Choice of the metric on the orthonormal space is motivated by the
    % symmetry present in the space. The metric on the positive definite
    % cone is its natural bi-invariant metric.
    % The result is equal to: trace( (X\eta) * (X\zeta) )
    M.inner = @innerproduct;
    function iproduct = innerproduct(X, eta, zeta)
        iproduct = 0;
        for kk = 1 : k
            iproduct = iproduct + trinner(X(:,:,kk)\eta(:,:,kk), X(:,:,kk)\zeta(:,:,kk));
        end
    end
    
    % Notice that X\eta is *not* symmetric in general.
    % The result is equal to: sqrt(trace((X\eta)^2))
    % There should be no need to take the real part, but rounding errors
    % may cause a small imaginary part to appear, so we discard it.
    M.norm = @innernorm;
    function inorm = innernorm(X, eta)
        inorm = 0;
        for kk = 1:k
            inorm = inorm + real(trnorm(X(:,:,kk)\eta(:,:,kk)))^2;
        end
        inorm = sqrt(inorm);
    end
    
    %     % Same here: X\Y is not symmetric in general.
    %     % Same remark about taking the real part.
    %     M.dist = @innerdistance;
    %     function idistance = innerdistance(X, Y)
    %         idistance = 0;
    %         for kk = 1:k
    %             idistance = idistance + real(trnorm(real(logm(X(:,:,kk)\Y(:,:,kk)))));
    %         end
    %     end
    
    M.typicaldist = @() sqrt(k*n*(n+1)/2); % BM: to be looked into.
    
    
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad)
        egradscaled = nan(size(egrad));
        for kk = 1:k
            egradscaled(:,:,kk) = X(:,:,kk)*symm(egrad(:,:,kk))*X(:,:,kk);
        end
        
        % Project onto the set X1dot + X2dot + ... = 0.
        % That is rgrad = Xk*egradk*Xk + Xk*Lambdasol*Xk
        rgrad = M.proj(X, egradscaled);
        
        %   % Debug
        %   norm(sum(rgrad,3), 'fro') % BM: this should be zero.
    end
    
    
    M.ehess2rhess = @ehess2rhess;
    function Hess = ehess2rhess(X, egrad, ehess, eta)
        Hess = nan(size(X));
        
        egradscaled = nan(size(egrad));
        egradscaleddot = nan(size(egrad));
        for kk = 1:k
            egradk = symm(egrad(:,:,kk));
            ehessk = symm(ehess(:,:,kk));
            Xk = X(:,:,kk);
            etak = eta(:,:,kk);

            egradscaled(:,:,kk) = Xk*egradk*Xk;
            egradscaleddot(:,:,kk) = Xk*ehessk*Xk + 2*symm(etak*egradk*Xk);
        end

        % Compute Lambdasol
        RHS = - sum(egradscaled,3);
        [Lambdasol] = mylinearsolve(X, RHS);


        % Compute Lambdasoldot
        temp = nan(size(egrad));;
        for kk = 1:k
            Xk = X(:,:,kk);
            etak = eta(:,:,kk);

            temp(:,:,kk) = 2*symm(etak*Lambdasol*Xk);
        end
        RHSdot = - sum(egradscaleddot,3) - sum(temp,3);
        [Lambdasoldot] = mylinearsolve(X, RHSdot);


        for kk = 1:k
            egradk = symm(egrad(:,:,kk));
            ehessk = symm(ehess(:,:,kk));
            Xk = X(:,:,kk);
            etak = eta(:,:,kk);

            % Directional derivatives of the Riemannian gradient
            % Note that Riemannian grdient is Xk*egradk*Xk + Xk*Lambdasol*Xk.
            % rhessk = Xk*(ehessk + Lambdasoldot)*Xk + 2*symm(etak*(egradk + Lambdasol)*Xk);
            % rhessk = rhessk - symm(etak*(egradk + Lambdasol)*Xk);
            rhessk = Xk*(ehessk + Lambdasoldot)*Xk + symm(etak*(egradk + Lambdasol)*Xk);

            Hess(:,:,kk) = rhessk;
        end
        
        % Project onto the set X1dot + X2dot + ... = 0.
        Hess = M.proj(X, Hess);
    end
    
    
    % Project onto the set X1dot + X2dot + ... = 0.
    M.proj = @innerprojection;
    function zeta = innerprojection(X, eta)
        % Project eta onto the set X1dot + X2dot + ... = 0.
        % Projected eta = eta + Xk*Lambdasol* Xk.

        RHS = - sum(eta,3);

        [Lambdasol] = mylinearsolve(X, RHS); % It solves sum Xi Lambdasol Xi = RHS.

        zeta = zeros(size(eta));
        for jj = 1 : k
            zeta(:,:,jj) = eta(:,:,jj) + X(:,:,jj)*Lambdasol*X(:,:,jj);
        end
        
        %   % Debug
        %   neta = eta - zeta; % Normal vector
        %   innerproduct(X, zeta, neta) % This should be zero
    end

    function [Lambdasol] = mylinearsolve(X, RHS)
        % Solve the linear system
        % sum Xi Lambdasol Xi = RHS.
        tol_omegax_pcg = 1e-8;
        max_iterations_pcg = 100;
        
        % vectorize RHS
        rhs = RHS(:);
        
        % Call PCG
        [lambdasol, ~, ~, ~] = pcg(@compute_matrix_system, rhs, tol_omegax_pcg, max_iterations_pcg);

        % Devectorize lambdasol.
        Lambdasol = symm(reshape(lambdasol, [n n]));
        
        function lhslambda = compute_matrix_system(lambda)
            Lambda = symm(reshape(lambda, [n n]));
            lhsLambda = zeros(n,n);
            for kk = 1:k
                lhsLambda = lhsLambda + X(:,:,kk)*Lambda*X(:,:,kk);
            end
            lhslambda = lhsLambda(:);
        end

        % % Debug
        % lhsLambda = zeros(n,n);
        % for kk = 1:k
        %     lhsLambda = lhsLambda + (X(:,:,kk)*Lambdasol*X(:,:,kk));
        % end
        % norm(lhsLambda - RHS, 'fro')/norm(RHS,'fro')
        % keyboard;
    end
    
    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;
    
    myeps = eps;
    
    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            teta = eta;
        else
            teta = t*eta;
        end
        % The symm() call is mathematically unnecessary but numerically
        % necessary.
        Y = nan(size(X));
        for kk=1:k
            % Second-order approximation of expm
            Y(:,:,kk) = symm(X(:,:,kk) + teta(:,:,kk) + .5*teta(:,:,kk)*((X(:,:,kk) + myeps*eye(n) )\teta(:,:,kk)));
            
            % expm
            %Y(:,:,kk) = symm(X(:,:,kk)*real(expm((X(:,:,kk)  + myeps*eye(n))\(teta(:,:,kk)))));
        end
        Ysum = sum(Y, 3);
        Ysumsqrt = sqrtm(Ysum);
        for kk=1:kk
            Y(:,:,kk) = symm((Ysumsqrt\Y(:,:,kk))/Ysumsqrt);
        end
        %   % Debug
        %   norm(sum(Y, 3) - eye(n), 'fro') % This should be zero
    end
    
    M.hash = @(X) ['z' hashmd5(X(:))];
    
    % Generate a random symmetric positive definite matrix following a
    % certain distribution. The particular choice of a distribution is of
    % course arbitrary, and specific applications might require different
    % ones.
    M.rand = @random;
    function X = random()
        X = nan(n,n,k);
        for kk = 1:k
            D = diag(1+rand(n, 1));
            [Q, R] = qr(randn(n)); %#ok
            X(:,:,kk) = Q*D*Q';
        end
        Xsum = sum(X, 3);
        Xsumsqrt = sqrtm(Xsum);
        for kk = 1 : k
            X(:,:,kk) = symm((Xsumsqrt\X(:,:,kk))/Xsumsqrt);
        end
    end
    
    % Generate a uniformly random unit-norm tangent vector at X.
    M.randvec = @randomvec;
    function eta = randomvec(X)
        eta = nan(size(X));
        for kk = 1:k
            eta(:,:,kk) = symm(randn(n));
        end
        eta = M.proj(X, eta);
        nrm = M.norm(X, eta);
        eta = eta / nrm;
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(X) zeros(n,n,k);
    
    % Poor man's transporter: exploit the fact that all tangent spaces
    % are the set of symmetric matrices, so that the identity is a sort of
    % transporter. It may perform poorly if the origin and target (X1
    % and X2) are far apart though. This should not be the case for typical
    % optimization algorithms, which perform small steps.
    M.transp = @(X1, X2, eta) M.proj(X2, eta);
    
    % vec and mat are not isometries, because of the unusual inner metric.
    M.vec = @(X, U) U(:);
    M.mat = @(X, u) reshape(u, n, n, k);
    M.vecmatareisometries = @() false;
    
end
