function M = fixedrankUZquotientfactory(m, n, k)
% Manifold of m-by-n matrices of rank k with quotient geometry.
%
% function M = fixedrankUZquotientfactory(m, n, k)
%
% This follows the quotient geometry described in the following paper:
% B. Mishra, G. Meyer, S. Bonnabel and R. Sepulchre
% "Fixed-rank matrix factorizations and Riemannian low-rank optimization",
% arXiv, 2012.
%
% Paper link: http://arxiv.org/abs/1209.0430
%
% A point X on the manifold is represented as a structure with two
% fields: U and Z. The matrices U (mxk) is orthonormal,
% while the matrix Z (nxk) is a full column-rank
% matrix.
%
% Tangent vectors are represented as a structure with two fields: U, Z.

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Dec. 30, 2012.
% Contributors: 
% Change log: 



    M.name = @() sprintf('UZ'' quotient manifold of %dx%d matrices of rank %d', m, n, k);
    
    M.dim = @() (m+n-k)*k;
    
    % The choice of the metric is motivated by symmetry and scale
    % invariance in the total space
    M.inner = @(X, eta, zeta) eta.U(:).'*zeta.U(:)  + trace( (X.Z'*X.Z)\(eta.Z'*zeta.Z) );
                                        
    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));
    
    M.dist = @(x, y) error('fixedrankUZquotientfactory.dist not implemented yet.');
    
    M.typicaldist = @() 10*k;
    
    skew = @(X) .5*(X-X');
    symm = @(X) .5*(X+X');
    stiefel_proj = @(U, H) H - U*symm(U'*H);
    M.proj = @projection;
    function etaproj = projection(X, eta)

        eta.U = stiefel_proj(X.U, eta.U); % On the tangent space
        ZtZ = X.Z'*X.Z;
        SS = ZtZ;
        AS1 = 2*ZtZ*skew(X.U'*eta.U)*ZtZ;
        AS2 = 2*skew(ZtZ*(X.Z'*eta.Z));
        AS  = skew(AS1 + AS2);
        
        Omega = nested_sylvester(SS,AS);
        
        etaproj.U = eta.U - X.U*Omega;
        etaproj.Z = eta.Z - X.Z*Omega;
        
    end
    
    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        Y.U = uf(X.U + t*eta.U);
        Y.Z = X.Z + t*eta.Z;
        
    end
    
    M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end

        Y = retraction(X, eta, t);
        warning('manopt:fixedrankUZquotientfactory:exp', ...
               ['Exponential for fixed rank ' ...
                'manifold not implemented yet. Used retraction instead.']);
    end

    M.hash = @(X) ['z' manopt.privatetools.hashmd5(...
                                            [X.U(:) ; X.Z(:)]  )];
    
    M.rand = @random;
    % Factors U lives on Stiefel manifold, hence we will reuse
    % its random generator.
    stiefelm = manopt.manifolds.stiefel.stiefelfactory(m, k);
    function X = random()
        X.U = stiefelm.rand();
        X.Z = randn(n, k);
    end
    
    M.randvec = @randomvec;
    function eta = randomvec(X)
        eta.U = randn(m, k);
        eta.Z = randn(n, k);
        eta = projection(X, eta);
        nrm = M.norm(X, eta);
        eta.U = eta.U / nrm;
        eta.Z = eta.Z / nrm;
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(X) struct('U', zeros(m, k),...
                                                         'Z', zeros(n, k));
    
    M.transp = @(x1, x2, d) projection(x2, d);

end

% Linear combination of tangent vectors
function d = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>

    if nargin == 3
        d.U = a1*d1.U;
        d.Z = a1*d1.Z;
    elseif nargin == 5
        d.U = a1*d1.U + a2*d2.U;
        d.Z = a1*d1.Z + a2*d2.Z;
    else
        error('Bad use of fixedrankUZquotientfactory.lincomb.');
    end

end

function A = uf(A)
    [L, unused, R] = svd(A, 0); %#ok
    A = L*R';
end

function omega = nested_sylvester(sym_mat, asym_mat)
% omega=nested_sylvester(sym_mat,asym_mat)
% This function solves the system of nested Sylvester equations: 
%
%     X*sym_mat + sym_mat*X = asym_mat
%     Omega*sym_mat+sym_mat*Omega = X
% Mishra, Meyer, Bonnabel and Sepulchre, 'Fixed-rank matrix factorizations and Riemannian low-rank optimization'

    % Uses built-in lyap function, but does not exploit the fact that it's
    % twice the same sym_mat matrix that comes into play.
    X = lyap(sym_mat, -asym_mat);
    omega = lyap(sym_mat, -X);

end



