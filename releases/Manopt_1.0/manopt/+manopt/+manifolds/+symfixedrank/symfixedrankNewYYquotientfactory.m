function M = symfixedrankNewYYquotientfactory(m, k)
% Manifold of m-by-m symmetric positive semidefinite matrices of rank k.
%
% function M = symfixedrankNewYYquotientfactory(m, k)
%
% The goemetry is a symmetric version of the geometry in the techreport,
% B.Mishra, K. Adithya Apuroop and R. Sepulchre,
% "A Riemannian geometry for low-rank matrix completion",
% arXiv, 2012.
%
% Paper link: http://arxiv.org/abs/1211.1550
%
% A point X on the manifold is represented as a structure with 1 field: Y. 
% The matrix Y (mxk) is a full column-rank matrix.

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Dec. 30, 2012.
% Contributors: 
% Change log: 



    M.name = @() sprintf('YY''optimized quotient manifold of %dx%d matrices of rank %d', m, k);
    
    M.dim = @() (m + m -k)*k;
    
    % Metric is motivated by symmetry and scaling invariance
    M.inner = @(X, eta, zeta) trace((X.Y'*X.Y)*(eta.Y'*zeta.Y) ) ;                                    
    
    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));
    
    M.dist = @(x, y) error('symfixedrankNewYYquotientfactory.dist not implemented yet.');
    
    M.typicaldist = @() m*k;
   
    M.proj = @projection;
    function etaproj = projection(X, eta)

        YtY = X.Y'*X.Y;
         
        Lambda =  (eta.Y'*X.Y) / YtY  -   YtY \ (X.Y'*eta.Y);
        Lambda = Lambda/2;
        
        etaproj.Y = eta.Y + X.Y*Lambda;
        
    end
    
    M.retr = @retraction;
    function Xnew = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        Xnew.Y = X.Y + t*eta.Y;
       
        
    end
    
    M.exp = @exponential;
    function Xnew = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end

        Xnew = retraction(X, eta, t);
        warning('manopt:symfixedrankNewYYquotientfactory:exp', ...
               ['Exponential for fixed rank ' ...
                'manifold not implemented yet. Used retraction instead.']);
    end

    M.hash = @(X) ['z' manopt.privatetools.hashmd5(...
                                            X.Y(:)  )];
    
    M.rand = @random;
    
    function X = random()
        X.Y = randn(m, k);
    end
    
    M.randvec = @randomvec;
    function eta = randomvec(X)
        eta.Y = randn(m, k);
        eta = projection(X, eta);
        nrm = M.norm(X, eta);
        eta.Y = eta.Y / nrm;
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(X) struct('Y', zeros(m, k));
    
    M.transp = @(x1, x2, d) projection(x2, d);

end

% Linear combination of tangent vectors
function d = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>

    if nargin == 3
        d.Y = a1*d1.Y;
    elseif nargin == 5
        d.Y = a1*d1.Y + a2*d2.Y;
    else
        error('Bad use of symfixedrankNewYYquotientfactory.lincomb.');
    end

end





