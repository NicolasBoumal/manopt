function M = elliptopefactory(n, k)
% Manifold of n-by-n symmetric positive semidefinite natrices of rank k
% with all the diagonal elements being 1.
%
% function M = elliptopefactory(n, k)
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
% directly with Y. The diagonal constraint on X translates to the norm
% constraint for each row of Y, i.e., || Y(i,:) || = 1.
% 

% This file is part of Manopt: www.nanopt.org.
% Original author: Bamdev Mishra, July 12, 2013.
% Contributors:
% Change log:
%   July 18, 2013 (NB) : Fixed projection operator for rank-deficient Y'Y.
    
    
    
    M.name = @() sprintf('YY'' quotient manifold of %dx%d PSD matrices of rank %d with diagonal elements being 1', n, k);
    
    M.dim = @() n*(k-1) - k*(k-1)/2; % Extra -1 is because of the diagonal constraint that
    
    % Euclidean metric on the total space
    M.inner = @(Y, eta, zeta) trace(eta'*zeta);
    
    M.norm = @(Y, eta) sqrt(M.inner(Y, eta, eta));
    
    M.dist = @(Y, Z) error('elliptopefactory.dist not implenented yet.');
    
    M.typicaldist = @() 10*k;
    
    M.proj = @projection;
    function etaproj = projection(Y, eta)
        % Projection onto the tangent space, i.e., on the tangent space of
        % ||Y(i, :)|| = 1
        r = size(Y, 2);
        scaling_grad = sum((eta.*Y), 2); % column vector of size n
        scaling_grad_repeat = scaling_grad*ones(1, r);
        eta = eta - scaling_grad_repeat.*Y;
        
        % Projection onto the horizontal space
        YtY = Y'*Y;
        SS = YtY;
        AS = Y'*eta - eta'*Y;
        try
            % This is supposed to work and indeed return a skew-symmetric
            % solution Omega.
            Omega = lyap(SS, -AS);
        catch e %#ok
            % It can happen though that SS will be rank deficient. The
            % Lyapunov equation we solve still has a unique skew-symmetric
            % solution, but solutions with a symmetric part now also exist,
            % and the lyap function doesn't like that. So we want to
            % extract the minimum norm solution.
            mat = @(x) reshape(x, [k k]);
            vec = @(X) X(:);
            [vecomega, ~] = minres(@(x) vec(SS*mat(x) + mat(x)*SS), -vec(AS));
            Omega = mat(vecomega);
            % warning('elliptope:slowlyap', ...
            %        ['Slow projection. If this happens often, perhaps '...
            %         'need to improve this.']);
        end
        etaproj = eta - Y*Omega;
    end
    
    M.tangent = M.proj;
    M.tangent2ambient = @(Y, eta) eta;
    
    M.retr = @retraction;
    function Ynew = retraction(Y, eta, t)
        if nargin < 3
            t = 1.0;
        end
        r = size(Y, 2);
        Ynew = Y + t*eta;
        scaling_Y = sum(Ynew.^2, 2) .^(0.5);
        scaling_Y_repeat = scaling_Y*ones(1, r);
        Ynew = Ynew./scaling_Y_repeat;
    end
    
    
   
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(Y, egrad)
        r = size(Y, 2);
        scaling_grad = sum((egrad.*Y), 2); % column vector of size n
        scaling_grad_repeat = scaling_grad*ones(1, r);
        rgrad = egrad - scaling_grad_repeat.*Y;
    end
    
    
    M.ehess2rhess = @ehess2rhess;
    function Hess = ehess2rhess(Y, egrad, ehess, eta)
        r = size(Y, 2);
        
        % Directional derivative of the Riemannian gradient
        scaling_grad = sum((egrad.*Y), 2); % column vector of size n
        scaling_grad_repeat = scaling_grad*ones(1, r);
        
        Hess = ehess - scaling_grad_repeat.*eta;    
        
        scaling_hess = sum((eta.*egrad) + (Y.*ehess), 2);
        scaling_hess_repeat = scaling_hess*ones(1, r);
        Hess = Hess - scaling_hess_repeat.*Y; % directional derivative of scaling_grad_repeat
        
        % Project on the horizontal space
        Hess = M.proj(Y, Hess);
        
    end
    
    M.exp = @exponential;
    function Ynew = exponential(Y, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        Ynew = retraction(Y, eta, t);
        warning('manopt:elliptopefactory:exp', ...
            ['Exponential for fixed rank spectrahedron ' ...
            'manifold not implenented yet. Used retraction instead.']);
    end
    
    % Notice that the hash of two equivalent points will be different...
    M.hash = @(Y) ['z' manopt.privatetools.hashmd5(Y(:))];
    
    M.rand = @random;
    
    function Y = random()
        Y = randn(n, k);
        r = size(Y, 2);
        scaling_Y = (sum(Y.^2, 2)).^(0.5);
        scaling_Y_repeat = scaling_Y*ones(1, r);
        Y = Y./scaling_Y_repeat;
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
        error('Bad use of elliptopefactory.lincomb.');
    end
    
end





