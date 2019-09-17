function M = symfixedrankpolarfactory(m, k)
% Manifold of symmetric, positive semidefinite matrices of size m and rank k with polar quotient geometry.
%
% function M = symfixedrankpolarfactory(m, k)
%
% The first-order geometry follows the balanced quotient geometry described
% in the paper,
% "Linear regression under fixed-rank constraints: a Riemannian approach",
% G. Meyer, S. Bonnabel and R. Sepulchre, ICML 2011.
%
% Paper link: http://www.icml-2011.org/papers/350_icmlpaper.pdf.
%
% The second-order geometry follows from the paper
% "Fixed-rank matrix factorizations and Riemannian low-rank optimization",
% B. Mishra, G. Meyer, S. Bonnabel and R. Sepulchre,
% Computational Statistics, 29(3 - 4), pp. 591 - 621, 2014.
%
% A point X on the manifold is represented as a structure with two
% fields: U and B. The matrix U (mxk) is orthonormal,
% while the matrix B (kxk) is a symmetric, positive definite.
%
% Tangent vectors are represented as a structure with two fields: U and B.
% These are the horizontal lifts of the tangent vectors on the quotient 
% manifold.
%
%
% For first-order geometry, please cite the Manopt paper as well as the research paper:
%     @InProceedings{meyer2011linear,
%       Title        = {Linear regression under fixed-rank constraints: a {R}iemannian approach},
%       Author       = {Meyer, G. and Bonnabel, S. and Sepulchre, R.},
%       Booktitle    = {{28th International Conference on Machine Learning}},
%       Year         = {2011},
%       Organization = {{ICML}}
%     }
% For second-order geometry, please cite the Manopt paper as well as the research paper:
%     @Article{mishra2014fixedrank,
%       Title   = {Fixed-rank matrix factorizations and {Riemannian} low-rank optimization},
%       Author  = {Mishra, B. and Meyer, G. and Bonnabel, S. and Sepulchre, R.},
%       Journal = {Computational Statistics},
%       Year    = {2014},
%       Number  = {3--4},
%       Pages   = {591--621},
%       Volume  = {29},
%       Doi     = {10.1007/s00180-013-0464-z}
%     }
%
%
% See also symfixedrankYYfactory sympositivedefinitefactory
    
    % This file is part of Manopt: www.manopt.org.
    % Original author: Bamdev Mishra, April 05, 2019.
    % Contributors:
    % Change log:
    %
    
    M.name = @() sprintf('UBU'' quotient manifold of %dx%d symmetric, positive semidefinite matrices of rank %d', m, m, k);
    
    M.dim = @() m*k - k*(k-1)/2;
    
    % Choice of the metric on the orthonormal space is motivated by the symmetry present in the
    % space. The metric on the positive definite space is its natural 
    % bi-invarint metric. The metric on the Stiefel manifold is the
    % standard Euclidean metric.
    trinner = @(A, B) A(:)'*B(:)
    M.inner = @inner;
    function ip = inner(X, eta, zeta) 
    	Binvetazeta = X.B \ [eta.B, zeta.B];
		ip = eta.U(:).'*zeta.U(:)  ...
				+ trinner(Binvetazeta(:, 1:k), Binvetazeta(:, 1+k : 2*k));
    end
    
    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));
    
    M.dist = @(x, y) error('symfixedrankpolarfactory.dist not implemented yet.');
    
    M.typicaldist = @() 10*k;
    
    skew = @(X) .5*(X-X');
    symm = @(X) .5*(X+X');
    stiefel_proj = @(U, H) H - U*symm(U'*H);
    
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad)
        rgrad.U = stiefel_proj(X.U, egrad.U);
        rgrad.B = X.B*symm(egrad.B)*X.B;
    end
    
    
    M.ehess2rhess = @ehess2rhess;
    function Hess = ehess2rhess(X, egrad, ehess, eta)
        
        % Riemannian gradient for the factor B.
        rgrad.B = X.B*symm(egrad.B)*X.B;
        
        % Directional derivatives of the Riemannian gradient.
        Hess.U = ehess.U - eta.U*symm(X.U'*egrad.U);
        Hess.U = stiefel_proj(X.U, Hess.U);
        
        Hess.B = X.B*symm(ehess.B)*X.B +  2*symm(eta.B*symm(egrad.B)*X.B);
        
        % Correction factor for the non-constant metric on the factor B.
        Hess.B = Hess.B - symm(eta.B*(X.B\rgrad.B));
        
        % Projection onto the horizontal space.
        Hess = M.proj(X, Hess);
    end
    
    
    M.proj = @projection;
    function etaproj = projection(X, eta)
        % First, projection onto the tangent space of the total space.
        eta.U = stiefel_proj(X.U, eta.U);
        eta.B = symm(eta.B);
        
        % Then, projection onto the horizontal space.
        SS = X.B*X.B;
        AS = X.B*(skew(X.U'*eta.U) - 2*skew(X.B\eta.B))*X.B;
        
        % Compute skew-symmetric Omega.
        % To solve the system SS*Omega + Omega*SS - B*Omega*B = AS
        Omega = mylinearsystem(SS, AS);
        
        %         % Debug: Omega is skew-symmetric.
        %         norm(Omega + Omega','fro')
        
        etaproj.U = eta.U - X.U*Omega;
        etaproj.B = eta.B - (X.B*Omega - Omega*X.B);
        
        %         % Debug
        %         neta.U = eta.U - etaproj.U;
        %         neta.B = eta.B - etaproj.B;
        %         M.inner(X, neta, etaproj)
    end
    
    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;
    
    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        B = X.B;
        tetaB = t*eta.B;
        
        % Another approach.
        % Y.B = symm(B*real(expm(B\(tetaB))));
        Y.B = symm(B + tetaB + .5*tetaB*(B\tetaB));
        
        Y.U = uf(X.U + t*eta.U);
    end
    
    
    M.hash = @(X) ['z' hashmd5([X.U(:) ; X.B(:)])];
    
    M.rand = @random;
    % Factor U is on Stiefel manifold, hence we reuse
    % its random generator.
    stiefelm = stiefelfactory(m, k);
    function X = random()
        X.U = stiefelm.rand();
        X.B = diag(1+rand(k, 1));
    end
    
    M.randvec = @randomvec;
    function eta = randomvec(X)
        % A random vector on the horizontal space.
        eta.U = randn(m, k);
        eta.B = randn(k, k);
        eta = projection(X, eta);
        nrm = M.norm(X, eta);
        eta.U = eta.U / nrm;
        eta.B = eta.B / nrm;
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(X) struct('U', zeros(m, k), 'B', zeros(k, k));
    
    M.transp = @(x1, x2, d) projection(x2, d);
    
    % vec and mat are not isometries, because of the scaled inner metric.
    M.vec = @(X, U) [U.U(:) ; U.B(:)];
    M.mat = @(X, u) struct('U', reshape(u(1:(m*k)), m, k), ...
        'B', reshape(u((m*k+1): end), k, k));
    M.vecmatareisometries = @() false;
    
end

% Linear combination of tangent vectors.
function d = lincomb(x, a1, d1, a2, d2) %#ok<INLSL>
    
    if nargin == 3
        d.U = a1*d1.U;
        d.B = a1*d1.B;
    elseif nargin == 5
        d.U = a1*d1.U + a2*d2.U;
        d.B = a1*d1.B + a2*d2.B;
    else
        error('Bad use of symfixedrankpolarfactory.lincomb.');
    end
    
end

function A = uf(A)
    [U, unused, R] = svd(A, 0); %#ok
    A = U*R';
end

function Omega = mylinearsystem(Bsq, RHS)
    % We want to sove the system Bsq*Omega + Omega*Bsq - B*Omega*B = RHS.
    [u, s2] = eig(Bsq);
    s2 = diag(s2);
    s = sqrt(s2);
    rhs = u'*RHS*u;
    e = ones(size(Bsq,1),1);% column vector of ones.
    Omega = u*(rhs./(e*s2' + s2*e' -s*s'))*u';
    
    %     % Debug
    %     B = u*diag(s)*u';
    %     norm(Bsq*Omega + Omega*Bsq - B*Omega*B - RHS,'fro')/norm(RHS,'fro')
end
