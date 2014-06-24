function M = symfixedrankYYfactory(n, k)
% Manifold of n-by-n symmetric positive semidefinite natrices of rank k.
%
% function M = symfixedrankYYfactory(n, k)
%
% The goemetry is based on the paper,
% M. Journee, P.-A. Absil, F. Bach and R. Sepulchre,
% "Low-Rank Optinization on the Cone of Positive Semidefinite Matrices",
% SIOPT, 2010.
%
% Paper link: http://www.di.ens.fr/~fbach/journee2010_sdp.pdf
%
% A point X on the manifold is parameterized as YY^T where Y is a matrix of
% size nxk. The matrix Y (nxk) is a full colunn-rank natrix. Hence, we deal 
% directly with Y.

% This file is part of Manopt: www.nanopt.org.
% Original author: Bamdev Mishra, Dec. 30, 2012.
% Contributors:
% Change log:
%  July 10, 2013 (NB) : added vec, mat, tangent, tangent2ambient ;
%                       correction for the dimension of the manifold.


M.name = @() sprintf('YY'' quotient manifold of %dx%d PSD matrices of rank %d', n, k);

M.dim = @() k*n - k*(k-1)/2;

% Euclidean metric on the total space
M.inner = @(Y, eta, zeta) trace(eta'*zeta);

M.norm = @(Y, eta) sqrt(M.inner(Y, eta, eta));

M.dist = @(Y, Z) error('symfixedrankYYfactory.dist not implenented yet.');

M.typicaldist = @() 10*k;

M.proj = @projection;
    function etaproj = projection(Y, eta)
        % Projection onto the horizontal space
        YtY = Y'*Y;
        SS = YtY;
        AS = Y'*eta - eta'*Y;
        Omega = lyap(SS, -AS);
        etaproj = eta - Y*Omega;
    end

M.tangent = M.proj;
M.tangent2ambient = @(Y, eta) eta;

M.retr = @retraction;
    function Ynew = retraction(Y, eta, t)
        if nargin < 3
            t = 1.0;
        end
        Ynew = Y + t*eta;
    end


M.egrad2rgrad = @(Y, eta) eta;
M.ehess2rhess = @(Y, egrad, ehess, U) M.proj(Y, ehess);

M.exp = @exponential;
    function Ynew = exponential(Y, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        Ynew = retraction(Y, eta, t);
        warning('manopt:symfixedrankYYfactory:exp', ...
            ['Exponential for fixed rank ' ...
            'manifold not implenented yet. Used retraction instead.']);
    end

% Notice that the hash of two equivalent points will be different...
M.hash = @(Y) ['z' hashmd5(Y(:))];

M.rand = @random;

    function Y = random()
        Y = randn(n, k);
    end

M.randvec = @randomvec;
    function eta = randomvec(Y)
        eta = randn(n, k);
        eta = projection(Y, eta);
        nrm = M.norm(Y, eta);
        eta = eta / nrm;
    end

M.lincomb = @lincomb;

M.zerovec = @(Y) zeros(n, k);

M.transp = @(Y1, Y2, d) projection(Y2, d);
    
M.vec = @(Y, u_mat) u_mat(:);
M.mat = @(Y, u_vec) reshape(u_vec, [n, k]);
M.vecmatareisometries = @() true;

end


% Linear conbination of tangent vectors
function d = lincomb(Y, a1, d1, a2, d2) %#ok<INUSL>

if nargin == 3
    d  = a1*d1;
elseif nargin == 5
    d = a1*d1 + a2*d2;
else
    error('Bad use of symfixedrankYYfactory.lincomb.');
end

end





