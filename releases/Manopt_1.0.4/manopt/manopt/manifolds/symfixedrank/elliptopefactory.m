function M = elliptopefactory(n, k)
% Manifold of n-by-n PSD matrices of rank k with unit diagonal elements.
%
% function M = elliptopefactory(n, k)
%
% The geometry is based on the paper,
% M. Journee, P.-A. Absil, F. Bach and R. Sepulchre,
% "Low-Rank Optimization on the Cone of Positive Semidefinite Matrices",
% SIOPT, 2010.
%
% Paper link: http://www.di.ens.fr/~fbach/journee2010_sdp.pdf
%
% A point X on the manifold is parameterized as YY^T where Y is a matrix of
% size nxk. The matrix Y (nxk) is a full column-rank matrix. Hence, we deal
% directly with Y. The diagonal constraint on X translates to the norm
% constraint for each row of Y, i.e., || Y(i, :) || = 1.
% 

% This file is part of Manopt: www.nanopt.org.
% Original author: Bamdev Mishra, July 12, 2013.
% Contributors:
% Change log:
%   July 18, 2013 (NB) : Fixed projection operator for rank-deficient Y'Y.
%   Aug.  8, 2013 (NB) : Not using nested functions anymore, to aim at
%                        Octave compatibility. Sign error in right hand
%                        side of the call to minres corrected.
    
    
    
    M.name = @() sprintf('YY'' quotient manifold of %dx%d PSD matrices of rank %d with diagonal elements being 1', n, k);
    
    M.dim = @() n*(k-1) - k*(k-1)/2; % Extra -1 is because of the diagonal constraint that
    
    % Euclidean metric on the total space
    M.inner = @(Y, eta, zeta) trace(eta'*zeta);
    
    M.norm = @(Y, eta) sqrt(M.inner(Y, eta, eta));
    
    M.dist = @(Y, Z) error('elliptopefactory.dist not implemented yet.');
    
    M.typicaldist = @() 10*k;
    
    M.proj = @projection;
    
    M.tangent = M.proj;
    M.tangent2ambient = @(Y, eta) eta;
    
    M.retr = @retraction;
    
    M.egrad2rgrad = @egrad2rgrad;
    
    M.ehess2rhess = @ehess2rhess;
    
    M.exp = @exponential;
    
    % Notice that the hash of two equivalent points will be different...
    M.hash = @(Y) ['z' hashmd5(Y(:))];
    
    M.rand = @() random(n, k);
    
    M.randvec = @randomvec;
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(Y) zeros(n, k);
    
    M.transp = @(Y1, Y2, d) projection(Y2, d);
    
    M.vec = @(Y, u_mat) u_mat(:);
    M.mat = @(Y, u_vec) reshape(u_vec, [n, k]);
    M.vecmatareisometries = @() true;
    
end


% Projection onto the tangent space, i.e., on the tangent space of
% ||Y(i, :)|| = 1
function etaproj = projection(Y, eta)
    [~, k] = size(Y);
    scaling_grad = sum((eta.*Y), 2); % column vector of size n
    scaling_grad_repeat = scaling_grad*ones(1, k);
    eta = eta - scaling_grad_repeat.*Y;

    % Projection onto the horizontal space
    YtY = Y'*Y;
    SS = YtY;
    AS = Y'*eta - eta'*Y;
    try
        % This is supposed to work and indeed return a skew-symmetric
        % solution Omega.
        Omega = lyap(SS, -AS);
    catch %#ok<CTCH> Octave does not handle the input of catch ... :(
        % It can happen though that SS will be rank deficient. The
        % Lyapunov equation we solve still has a unique skew-symmetric
        % solution, but solutions with a symmetric part now also exist,
        % and the lyap function doesn't like that. So we want to
        % extract the minimum norm solution.
        mat = @(x) reshape(x, [k k]);
        vec = @(X) X(:);
        is_octave = exist('OCTAVE_VERSION', 'builtin');
        if ~is_octave
            [vecomega, ~] = minres(@(x) vec(SS*mat(x) + mat(x)*SS), vec(AS));
        else
            [vecomega, ~] = gmres(@(x) vec(SS*mat(x) + mat(x)*SS), vec(AS));
        end
        Omega = mat(vecomega);
        % warning('elliptope:slowlyap', ...
        %        ['Slow projection. If this happens often, perhaps '...
        %         'need to improve this.']);
    end
    % % Make sure the result is skew-symmetric (does not seem necessary).
    % Omega = (Omega-Omega')/2;
    etaproj = eta - Y*Omega;
end

% Retraction
function Ynew = retraction(Y, eta, t)
    if nargin < 3
        t = 1.0;
    end
    k = size(Y, 2);
    Ynew = Y + t*eta;
    scaling_Y = sum(Ynew.^2, 2) .^(0.5);
    scaling_Y_repeat = scaling_Y*ones(1, k);
    Ynew = Ynew./scaling_Y_repeat;
end

% Exponential map
function Ynew = exponential(Y, eta, t)
    if nargin < 3
        t = 1.0;
    end

    Ynew = retraction(Y, eta, t);
    warning('manopt:elliptopefactory:exp', ...
        ['Exponential for fixed rank spectrahedron ' ...
        'manifold not implenented yet. Used retraction instead.']);
end

% Euclidean gradient to Riemannian gradient conversion
function rgrad = egrad2rgrad(Y, egrad)
    k = size(Y, 2);
    scaling_grad = sum((egrad.*Y), 2); % column vector of size n
    scaling_grad_repeat = scaling_grad*ones(1, k);
    rgrad = egrad - scaling_grad_repeat.*Y;
end

% Euclidean Hessian to Riemannian Hessian conversion
function Hess = ehess2rhess(Y, egrad, ehess, eta)
    k = size(Y, 2);

    % Directional derivative of the Riemannian gradient
    scaling_grad = sum((egrad.*Y), 2); % column vector of size n
    scaling_grad_repeat = scaling_grad*ones(1, k);

    Hess = ehess - scaling_grad_repeat.*eta;    

    scaling_hess = sum((eta.*egrad) + (Y.*ehess), 2);
    scaling_hess_repeat = scaling_hess*ones(1, k);
    % directional derivative of scaling_grad_repeat
    Hess = Hess - scaling_hess_repeat.*Y;

    % Project on the horizontal space
    Hess = projection(Y, Hess);
end

% Random point generation on the manifold
function Y = random(n, k)
    Y = randn(n, k);
    scaling_Y = (sum(Y.^2, 2)).^(0.5);
    scaling_Y_repeat = scaling_Y*ones(1, k);
    Y = Y./scaling_Y_repeat;
end

% Random vector generation at Y
function eta = randomvec(Y)
    eta = randn(size(Y));
    eta = projection(Y, eta);
    nrm = norm(eta, 'fro');
    eta = eta / nrm;
end

% Linear conbination of tangent vectors
function d = lincomb(Y, a1, d1, a2, d2) %#ok<INUSL>
    
    if nargin == 3
        d  = a1*d1;
    elseif nargin == 5
        d = a1*d1 + a2*d2;
    else
        error('Bad use of elliptopefactory.lincomb.');
    end
    
end
