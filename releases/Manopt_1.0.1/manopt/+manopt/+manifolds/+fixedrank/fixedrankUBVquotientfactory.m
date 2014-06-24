function M = fixedrankUBVquotientfactory(m, n, k)
% Manifold of m-by-n matrices of rank k with polar quotient geometry.
%
% function M = fixedrankUBVquotientfactory(m, n, k)
%
% Follows the polar quotient geometry described in the following paper:
% G. Meyer, S. Bonnabel and R. Sepulchre,
% "Linear regression under fixed-rank constraints: a Riemannian approach",
% ICML 2011.
%
% Paper link: http://www.icml-2011.org/papers/350_icmlpaper.pdf
%
% A point X on the manifold is represented as a structure with three
% fields: U, B and V. The matrices U (mxk) and V (nxk) are orthonormal,
% while the matrix B (kxk) is a symmetric positive definite full rank
% matrix.
%
% Tangent vectors are represented as a structure with three fields: U, B
% and V.

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Dec. 30, 2012.
% Contributors: 
% Change log: 



M.name = @() sprintf('UBV'' quotient manifold of %dx%d matrices of rank %d', m, n, k);

M.dim = @() (m+n-k)*k;

% Choice of the metric on the orthnormal space is motivated by the symmetry present in the
% space. The metric on the positive definite space is its natural metric.
M.inner = @(X, eta, zeta) eta.U(:).'*zeta.U(:) + eta.V(:).'*zeta.V(:) ...
    + trace( (X.B\eta.B) * (X.B\zeta.B) );

M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));

M.dist = @(x, y) error('fixedrankUBVquotientfactory.dist not implemented yet.');

M.typicaldist = @() 10*k;

skew = @(X) .5*(X-X');
symm = @(X) .5*(X+X');
stiefel_proj = @(U, H) H - U*symm(U'*H);
M.proj = @projection;
    function etaproj = projection(X, eta)

        % Should not be here
% %         % If eta is a matrix of size mxn, turn it into a structure with
% %         % fields U, B, V.
% %         if ~isstruct(eta)
% %             eta_struct.U = eta*X.V*X.B;
% %             eta_struct.B = X.B*X.U'*eta*X.V*X.B;
% %             eta_struct.V = eta'*X.U*X.B;
% %             eta = eta_struct;
% %         end
        
        % First, projection onto the tangent space of the total sapce
        eta.U = stiefel_proj(X.U, eta.U);
        eta.V = stiefel_proj(X.V, eta.V);
        eta.B = symm(eta.B);
        
        % Then, projection onto the horizontal space
        SS = X.B*X.B;
        AS = X.B*(skew(X.U'*eta.U) + skew(X.V'*eta.V) - 2*skew(X.B\eta.B))*X.B;
%         omega = sylvester(SS, AS);
        omega = lyap(SS, -AS);

        etaproj.U = eta.U - X.U*omega;
        etaproj.B = eta.B - (X.B*omega - omega*X.B);
        etaproj.V = eta.V - X.V*omega;
        
    end

M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        L = chol(X.B);
        Y.B = L'*expm(L'\(t*eta.B)/L)*L;
        Y.U = uf(X.U + t*eta.U);
        Y.V = uf(X.V + t*eta.V);
        
    end

M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        Y = retraction(X, eta, t);
        warning('manopt:fixedrankUBVquotientfactory:exp', ...
            ['Exponential for fixed rank ' ...
            'manifold not implemented yet. Used retraction instead.']);
    end

M.hash = @(X) ['z' manopt.privatetools.hashmd5(...
    [X.U(:) ; X.B(:) ; X.V(:)]  )];

M.rand = @random;
% Factors U and V live on Stiefel manifolds, hence we will reuse
% their random generator.
stiefelm = manopt.manifolds.stiefel.stiefelfactory(m, k);
stiefeln = manopt.manifolds.stiefel.stiefelfactory(n, k);
    function X = random()
        X.U = stiefelm.rand();
        X.V = stiefeln.rand();
        X.B = diag(1+rand(k, 1));
    end

M.randvec = @randomvec;
    function eta = randomvec(X)
        % A random vector on the horizontal space
        eta.U = randn(m, k);
        eta.V = randn(n, k);
        eta.B = randn(k, k);
        eta = projection(X, eta);
        nrm = M.norm(X, eta);
        eta.U = eta.U / nrm;
        eta.V = eta.V / nrm;
        eta.B = eta.B / nrm;
    end

M.lincomb = @lincomb;

M.zerovec = @(X) struct('U', zeros(m, k), 'B', zeros(k, k), ...
    'V', zeros(n, k));

M.transp = @(x1, x2, d) projection(x2, d);

end

% Linear combination of tangent vectors
function d = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>

if nargin == 3
    d.U = a1*d1.U;
    d.V = a1*d1.V;
    d.B = a1*d1.B;
elseif nargin == 5
    d.U = a1*d1.U + a2*d2.U;
    d.V = a1*d1.V + a2*d2.V;
    d.B = a1*d1.B + a2*d2.B;
else
    error('Bad use of fixedrankUBVquotientfactory.lincomb.');
end

end

function A = uf(A)
[L, unused, R] = svd(A, 0); %#ok
A = L*R';
end

% function omega = sylvester(sym_mat, asym_mat)
% % omega=sylvester(sym_mat,asym_mat)
% % This function solves the Sylvester equation: Omega*sym_mat+sym_mat*Omega=asym_mat
% %
% % Reference:
% % M. Journ?e, F. Bach, P.-A. Absil and R. Sepulchre, Low-rank optimization for semidefinite convex problems, arXiv:0807.4423v1, 2008
% 
% p=size(sym_mat,2);
% 
% [V,D]=eig(full(sym_mat));
% 
% O=zeros(p,p);
% 
% O2=V'*asym_mat*V;
% 
% for i=1:p-1,
%     for j=i+1:p,
%         A=D(i,i)+D(j,j);
%         if A~=0,
%             O(i,j)=O2(i,j)/A;
%         else
%             O(i,j)=0;
%         end
%     end
% end
% 
% % if sum(sum(isnan(O)))>0,
% %     sym_mat
% %     asym_mat
% %     O
% %     O2
% %     V
% %     D
% %     return
% % end
% 
% O = O-O';
% omega = V*O*V';
% 
% end
