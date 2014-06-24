function M = fixedrankGHquotientfactory(m, n, k)
% Manifold of m-by-n matrices of rank k with balanced quotient geometry.
%
% function M = fixedrankGHquotientfactory(m, n, k)
%
% This follows the balanced quotient geometry described in the following paper:
% G. Meyer, S. Bonnabel and R. Sepulchre,
% "Linear regression under fixed-rank constraints: a Riemannian approach",
% ICML 2011.
%
% Paper link: http://www.icml-2011.org/papers/350_icmlpaper.pdf
% 
% A point X on the manifold is represented as a structure with two
% fields: G and H. The matrices G (mxk) and H (nxk) are full column-rank
% matrices. 
% 
% Tangent vectors are represented as a structure with two fields: G, H

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Dec. 30, 2012.
% Contributors: 
% Change log: 



    M.name = @() sprintf('GH'' quotient manifold of %dx%d matrices of rank %d', m, n, k);
    
    M.dim = @() (m+n-k)*k;
    
    % Choice of the metric is motivated by the symmetry present in the space
    M.inner = @(X, eta, zeta) trace((X.G'*X.G)\(eta.G'*zeta.G)) + trace( (X.H'*X.H)\(eta.H'*zeta.H));
    
    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));
    
    M.dist = @(x, y) error('fixedrankGHquotientfactory.dist not implemented yet.');
    
    M.typicaldist = @() 10*k;
   
    M.proj = @projection;
    function etaproj = projection(X, eta)
        % Projection of the vector eta onto the horizontal space
        GtG = X.G'*X.G;
        HtH = X.H'*X.H;
        SS = (GtG)*(HtH);           
        AS = (GtG)*(X.H'*eta.H) - (eta.G'*X.G)*(HtH);  
        
        Omega = lyap(SS, SS,-AS);
        
        etaproj.G = eta.G + X.G*Omega';
        etaproj.H = eta.H - X.H*Omega; 
        
    end
    
    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        Y.G = X.G + t*eta.G;
        Y.H = X.H + t*eta.H;
        
        % Numerical conditioning step: A simpler version.
        % We need to ensure that G and H are do not have very relative
        % skewed norms.
        
        scaling = norm(X.G, 'fro')/norm(X.H, 'fro');
        scaling = sqrt(scaling);
        Y.G = Y.G / scaling;
        Y.H = Y.H * scaling;        
        
    end
    
    M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end

        Y = retraction(X, eta, t);
        warning('manopt:fixedrankGHquotientfactory:exp', ...
               ['Exponential for fixed rank ' ...
                'manifold not implemented yet. Used retraction instead.']);
    end

    M.hash = @(X) ['z' manopt.privatetools.hashmd5(...
                                            [X.G(:) ; X.H(:)]  )];
    
    M.rand = @random;
    
    function X = random()
        % A random point on the total space
        X.G = randn(m, k);
        X.H = randn(n, k);
    end
    
    M.randvec = @randomvec;
    function eta = randomvec(X)
        % A random vector in the horizontal space
        eta.G = randn(m, k);
        eta.H = randn(n, k);
        eta = projection(X, eta);
        nrm = M.norm(X, eta);
        eta.G = eta.G / nrm;
        eta.H = eta.H / nrm;
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(X) struct('G', zeros(m, k),'H', zeros(n, k));
    
    M.transp = @(x1, x2, d) projection(x2, d);

end

% Linear combination of tangent vectors
function d = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>

    if nargin == 3
        d.G = a1*d1.G;
        d.H = a1*d1.H;
    elseif nargin == 5
        d.G = a1*d1.G + a2*d2.G;
        d.H = a1*d1.H + a2*d2.H;
    else
        error('Bad use of fixedrankGHquotientfactory.lincomb.');
    end

end





